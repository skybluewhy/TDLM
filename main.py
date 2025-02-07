import os
import random
import numpy as np
import argparse
import torch
from data_loader import mydataset
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig
from models.modeling_roberta import RobertaForMaskedLM
import diffusion_word_freq
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
import fastNLP
from tqdm import tqdm
from sample import Categorical
import math


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
    parser.add_argument("--lr", default=1e-4, type=float, required=False)
    parser.add_argument("--epochs", default=10, type=int, required=False)
    parser.add_argument("--batch_size", default=1024, type=int, required=False)
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
        dim=22,  # vocab size
        schedule=diffusion_schedule,
        mask_id=21,  # MASK token id
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

    optimizer = AdamW(model.parameters(), lr=args.lr)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda n: n / 10000. + 1e-3 if n < 10000 else 100. / math.sqrt(n))

    train_dataset = mydataset("./data/11M_dataset.csv")

    logger = fastNLP.logger
    print('# of train data: {}'.format(len(train_dataset)))

    def collate_fn(batch_input):
        input_ids = [torch.tensor(d[0]) for d in batch_input]
        attention_mask = [torch.tensor(d[1]) for d in batch_input]
        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=0)

    model.train()

    # cls = torch.full((1, 1), fill_value=1, device=device)  # cls token id: 1
    # sep = torch.full((1, 1), fill_value=2, device=device)  # SEP token id: 2

    # att_ones = torch.ones((1, 1), device=device)
    # att_zeros = torch.zeros((1, 1), device=device)

    if args.timestep == 'none':
        def denoise_fn(targets, timestep, attention_mask):
            assert len(targets.size()) == 2  # bsz * seqlen
            # bsz = targets.size(0)
            # targets = torch.cat((cls.repeat(bsz, 1), targets, sep.repeat(bsz, 1)), dim=1)
            # attention_mask = torch.cat((att_ones.repeat(bsz, 1), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
            return model(input_ids=targets, timestep=timestep - 1, attention_mask=attention_mask)['logits']
    else:
        raise NotImplementedError

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    train_loss = .0
    nan_count = 0
    min_loss = 100
    for epoch in range(args.epochs):
        for i, batch in enumerate(tqdm(train_loader), args.load_step + 1):
            metrics = diffusion_word_freq.compute_kl_reverse_process(
                batch['input_ids'].to(device),
                diffusion_instance.sample_t(),
                denoise_fn=denoise_fn,
                diffusion=diffusion_instance,
                target_mask=batch['attention_mask'].to(device),
                hybrid_lambda=args.hybrid_lambda,
                predict_x0=args.predict_x0,
                word_freq_logits=None,
            )

            loss = metrics['loss'] / args.batch_size
            loss_list = [loss]
            if torch.stack(loss_list).isnan().any():
                nan_count += 1
                logger.warning(f'NaN encountered {nan_count} times')
                continue
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            optimizer.step()
            model.zero_grad()
            optimizer.zero_grad()
            warmup_scheduler.step()

            if i % args.logging_steps == args.logging_steps - 1:
                logger.info(f'Loss at step {i} is {train_loss / args.logging_steps}')
                cur_loss = train_loss / args.logging_steps
                if cur_loss < min_loss:
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'warmup_scheduler': warmup_scheduler.state_dict(),
                    }, f'./{save_path}/model' + str(epoch) + '.th')
                    min_loss = cur_loss

                train_loss = .0
