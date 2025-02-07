import os
import random
import numpy as np
import argparse
import torch
from data_loader import mydataset_eval
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig
from models.modeling_roberta import RobertaForMaskedLM
import diffusion_word_freq
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
import fastNLP
from tqdm import tqdm
from sample import Categorical
import math
import diffusion_word_freq as diffusion
import functools
import json
with open("./bert_model/vocab.json", 'r', encoding='UTF-8') as f:
    token_dict = json.load(f)
token_dict = dict(zip(token_dict.values(), token_dict.keys()))

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str, required=False)
    parser.add_argument("--task_name", default='lm1b', type=str, required=False)
    parser.add_argument("--lr", default=2e-5, type=float, required=False)
    parser.add_argument("--epochs", default=5, type=int, required=False)
    parser.add_argument("--batch_size", default=4, type=int, required=False)
    parser.add_argument("--word_freq_lambda", default=0.3, type=float, required=False)
    parser.add_argument("--num_steps", default=512, type=int, required=False)
    parser.add_argument("--eval_step_size", default=4, type=int, required=False)
    parser.add_argument("--hybrid_lambda", default=1e-2, type=float, required=False)
    parser.add_argument("--seed", default=42, type=int, required=False)
    parser.add_argument("--logging_steps", default=1000, type=int, required=False)
    parser.add_argument('--predict_x0', default=True, type=bool, required=False)
    parser.add_argument("--load_step", default=-1, type=int, required=False)
    parser.add_argument("--sample_strategy", default='Categorical', type=str, required=False)
    parser.add_argument("--schedule", default='mutual', type=str, required=False)
    parser.add_argument("--from_scratch", default=True, type=bool, required=False)
    parser.add_argument("--timestep", default='none', type=str, required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set_seed(args)
    if args.timestep in ['none', 'token']:
        from models.modeling_bert import BertForMaskedLM
    elif args.timestep == 'layerwise':
        from models.modeling_bert_new_timestep import BertForMaskedLM
    else:
        raise NotImplementedError

    log_dir = './logs'
    save_path = "checkpoint"

    if args.model_name_or_path in ['bert-base-uncased', 'bert-large-uncased']:
        model_cls = BertForMaskedLM
        cfg_cls = BertConfig
        tok_cls = BertTokenizer
    else:
        raise NotImplementedError

    sample_cls = Categorical()

    diffusion_schedule = diffusion_word_freq.create_discrete_diffusion_schedule(args.schedule, num_steps=args.num_steps)
    diffusion_instance = diffusion_word_freq.MaskDiffusion(
        dim=24,  # vocab size
        schedule=diffusion_schedule,
        mask_id=3,  # MASK token id
        sample_cls=sample_cls,
        word_freq_lambda=args.word_freq_lambda,
        device=device
    )

    if args.load_step > 0:
        ckpt = torch.load(os.path.join(save_path, f'{args.load_step}.th'))
    cfg = cfg_cls.from_pretrained("./bert_model")
    cfg.overall_timestep = diffusion_instance.num_steps

    if args.from_scratch:
        model = model_cls(cfg).to(device)
    elif args.load_step <= 0:
        model = model_cls.from_pretrained(args.model_name_or_path, config=cfg).to(device)
    else:
        model = model_cls(cfg).to(device)
        model.load_state_dict(ckpt['model'])
    ckpt = torch.load("./checkpoint/model.th", map_location=device)
    model.load_state_dict(ckpt['model'])

    test_dataset = mydataset_eval("./data/11M_dataset.csv")

    logger = fastNLP.logger
    print('# of test data: {}'.format(len(test_dataset)))

    def collate_fn(batch_input):
        input_ids = [torch.tensor(d[0]) for d in batch_input]
        attention_mask = [torch.tensor(d[1]) for d in batch_input]
        target_mask = [torch.tensor(d[2]) for d in batch_input]
        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        target_mask = pad_sequence(target_mask, batch_first=True)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask,
        }

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=0)

    model.eval()

    cls = torch.full((1, 1), fill_value=1, device=device)  # cls token id: 1
    sep = torch.full((1, 1), fill_value=2, device=device)  # SEP token id: 2

    att_ones = torch.ones((1, 1), device=device)
    att_zeros = torch.zeros((1, 1), device=device)

    if args.timestep == 'none':
        def denoise_fn(targets, timestep, attention_mask):
            assert len(targets.size()) == 2  # bsz * seqlen
            bsz = targets.size(0)
            targets = torch.cat((cls.repeat(bsz, 1), targets, sep.repeat(bsz, 1)), dim=1)
            attention_mask = torch.cat((att_ones.repeat(bsz, 1), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
            return model(input_ids=targets, timestep=timestep - 1, attention_mask=attention_mask)['logits'][:, 1:-1, :]
    else:
        raise NotImplementedError

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    with open(f'./generation_results/results.txt', 'w+') as f_raw:
        s_bleu = 0.
        dist_1 = 0.
        div_4 = 0.
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                for k, v in batch.items():
                    if v != None:
                        batch[k] = v.to(device)

                state = diffusion.discrete_diffusion_predict_fn(
                    ori_input=batch['input_ids'],
                    att_mask=batch['attention_mask'],
                    target_mask=batch['target_mask'],
                    denoise_fn=denoise_fn,
                    diffusion=diffusion_instance,
                    predict_x0=True,
                    sample_cls=sample_cls,
                    step_size=4,
                    topk=5,
                    show_process=False,
                    temperature=1.0,
                )['final_state']
                tcr_sequences = []
                target_att = batch['attention_mask'].cpu().numpy().tolist()
                cnt = 0
                for pred in state:
                    # sentence = pred.cpu().numpy().tolist()
                    sentence = pred.tolist()
                    cur_tcr_seq = ""
                    cur_target = target_att[cnt]
                    cnt += 1
                    num = -1
                    for s in sentence:
                        num += 1
                        if cur_target[num] == 1:
                            cur_tcr_seq += token_dict[s]
                    tcr_sequences.append(cur_tcr_seq)
                try:
                    # sentences = ["".join(pred[:pred.index('SEP')]) for pred in sentences]
                    sentences = []
                    for i in range(len(tcr_sequences)):
                        sentences.append(tcr_sequences[i])
                    print('\n'.join(sentences), file=f_raw, flush=True)
                    # print("complete " + str(i))
                except ValueError:
                    pass

