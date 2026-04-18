import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import random
import json
from transformers import AutoTokenizer
import torch
from torch.utils.data import Subset, DataLoader
from torch.optim import Adam
import torch.nn as nn
import datasets
from tqdm import tqdm
import argparse
import swanlab  # 替换 wandb

from eval import evaluate
import utils


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--data', type=str, default='WebOfScience')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--early-stop', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--update', type=int, default=1)
    parser.add_argument('--model', type=str, default='prompt')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--arch', type=str, default='bert-base-uncased')
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--graph', type=str, default='GAT')
    parser.add_argument('--low-res', default=False, action='store_true')
    parser.add_argument('--seed', default=3, type=int)
    
    # === 新增消融实验参数 ===
    parser.add_argument('--ablation_logits_mask', action='store_true', help='开启推理期向下对数几率掩码 (标签一致性)')
    parser.add_argument('--ablation_hierarchical_loss', action='store_true', help='开启三元组层次分离损失 (标签一致性)')
    parser.add_argument('--ablation_cross_attn', action='store_true', help='开启文本-标签概念深度交叉注意力 (语义交互)')
    parser.add_argument('--ablation_deep_prefix', action='store_true', help='开启SPIRIT深度前缀融合 (结构融合)')
    
    parser.add_argument('--ablation_mode', type=str, default='D', choices=['A', 'B', 'C', 'D'], 
                        help='A: 仅父节点排斥 | B: A+兄弟排斥 | C: B+密度自适应Margin | D: C+Focal加权(完全体)')
    
    return parser


class Save:
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

    def __call__(self, score, best_score, name):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save({'param': model_to_save.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'sche': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'score': score, 'args': self.args,
                    'best_score': best_score},
                   name)


if __name__ == '__main__':
    parser = parse()
    args, unknown = parser.parse_known_args()

    if args.wandb:
        swanlab.login(api_key='DQ5WCNvv4ra2WikI9c59a', save=False)
        swanlab.init(config=vars(args), project='bs', name=args.name)
    print(args)
    utils.seed_torch(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.arch)

    if os.path.exists('/kaggle/working'):
        WORKING_DIR = '/kaggle/working'
        KAGGLE_INPUT_DIR = '/kaggle/input/datasets/slbao8/bsdata/data'
        data_path = os.path.join(KAGGLE_INPUT_DIR, args.data)
    else:
        WORKING_DIR = current_dir
        data_path = os.path.join(current_dir, 'data', args.data)

    working_data_path = os.path.join(WORKING_DIR, 'data_cache', args.data)
    checkpoints_dir = os.path.join(WORKING_DIR, 'checkpoints')

    os.makedirs(working_data_path, exist_ok=True)
    os.makedirs(os.path.join(checkpoints_dir, args.data + '-' + args.name), exist_ok=True)

    args.name = args.data + '-' + args.name
    batch_size = args.batch

    label_dict = torch.load(os.path.join(data_path, 'value_dict.pt'), weights_only=False)
    label_dict = {i: v for i, v in label_dict.items()}

    slot2value = torch.load(os.path.join(data_path, 'slot.pt'), weights_only=False)
    value2slot = {}
    num_class = 0
    for s in slot2value:
        for v in slot2value[s]:
            value2slot[v] = s
            if num_class < v:
                num_class = v
    num_class += 1
    path_list = [(i, v) for v, i in value2slot.items()]
    for i in range(num_class):
        if i not in value2slot:
            value2slot[i] = -1

    def get_depth(x):
        depth = 0
        while value2slot[x] != -1:
            depth += 1
            x = value2slot[x]
        return depth

    depth_dict = {i: get_depth(i) for i in range(num_class)}
    max_depth = depth_dict[max(depth_dict, key=depth_dict.get)] + 1
    depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(max_depth)}

    for depth in depth2label:
        for l in depth2label[depth]:
            path_list.append((num_class + depth, l))

    if args.model == 'prompt':
        prompt_cache_path = os.path.join(working_data_path, args.model)
        if os.path.exists(prompt_cache_path):
            dataset = datasets.load_from_disk(prompt_cache_path)
        else:
            dataset = datasets.load_dataset('json',
                                            data_files={'train': os.path.join(data_path, f'{args.data}_train.json'),
                                                        'dev': os.path.join(data_path, f'{args.data}_dev.json'),
                                                        'test': os.path.join(data_path, f'{args.data}_test.json')})

            prefix = []
            for i in range(max_depth):
                prefix.append(tokenizer.vocab_size + num_class + i)
                prefix.append(tokenizer.vocab_size + num_class + max_depth)
            prefix.append(tokenizer.sep_token_id)

            def data_map_function(batch, tokenizer):
                new_batch = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'labels': []}
                for l, t in zip(batch['label'], batch['token']):
                    new_batch['labels'].append([[-100 for _ in range(num_class)] for _ in range(max_depth)])
                    for d in range(max_depth):
                        for i in depth2label[d]:
                            new_batch['labels'][-1][d][i] = 0
                        for i in l:
                            if new_batch['labels'][-1][d][i] == 0:
                                new_batch['labels'][-1][d][i] = 1
                    new_batch['labels'][-1] = [x for y in new_batch['labels'][-1] for x in y]

                    tokens = tokenizer(t, truncation=True)
                    new_batch['input_ids'].append(tokens['input_ids'][:-1][:512 - len(prefix)] + prefix)
                    new_batch['input_ids'][-1].extend(
                        [tokenizer.pad_token_id] * (512 - len(new_batch['input_ids'][-1])))
                    new_batch['attention_mask'].append(
                        tokens['attention_mask'][:-1][:512 - len(prefix)] + [1] * len(prefix))
                    new_batch['attention_mask'][-1].extend([0] * (512 - len(new_batch['attention_mask'][-1])))
                    new_batch['token_type_ids'].append([0] * 512)

                return new_batch

            dataset = dataset.map(lambda x: data_map_function(x, tokenizer), batched=True)
            dataset.save_to_disk(prompt_cache_path)

        dataset['train'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])
        dataset['dev'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])
        dataset['test'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])

        from models.prompt import Prompt

    else:
        raise NotImplementedError

    if args.low_res:
        low_res_path = os.path.join(working_data_path, 'low.json')
        if os.path.exists(low_res_path):
            index = json.load(open(low_res_path, 'r', encoding='utf-8'))
        else:
            index = [i for i in range(len(dataset['train']))]
            random.shuffle(index)
            json.dump(index, open(low_res_path, 'w', encoding='utf-8'))
        dataset['train'] = dataset['train'].select(index[len(index) // 5:len(index) // 10 * 3])

    # === 初始化模型并传入消融参数与拓扑 ===
    model = Prompt.from_pretrained(args.arch, num_labels=len(label_dict), path_list=path_list, layer=args.layer,
                                   graph_type=args.graph, data_path=data_path, depth2label=depth2label,
                                   value2slot=value2slot,
                                   ablation_logits_mask=args.ablation_logits_mask,
                                   ablation_hierarchical_loss=args.ablation_hierarchical_loss,
                                   ablation_cross_attn=args.ablation_cross_attn,
                                   ablation_deep_prefix=args.ablation_deep_prefix,
                                   ablation_mode=args.ablation_mode)
    model.init_embedding()
    model.to('cuda')

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    train = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    dev = DataLoader(dataset['dev'], batch_size=8, shuffle=False)
    optimizer = Adam(model.parameters(), lr=args.lr)

    save = Save(model, optimizer, None, args)
    best_score_macro = 0
    best_score_micro = 0
    early_stop_count = 0
    update_step = 0
    loss = 0

    for epoch in range(1000):
        if early_stop_count >= args.early_stop:
            print("Early stop!")
            break

        model.train()
        with tqdm(train) as p_bar:
            for batch in p_bar:
                batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output = model(**batch)

                loss_val = output['loss'].mean() if torch.cuda.device_count() > 1 else output['loss']
                loss_val.backward()
                loss += loss_val.item()
                update_step += 1
                if update_step % args.update == 0:
                    if args.wandb:
                        swanlab.log({'loss': loss, })
                    p_bar.set_description('loss:{:.4f}'.format(loss, ))
                    optimizer.step()
                    optimizer.zero_grad()
                    loss = 0
                    update_step = 0

        model.eval()
        pred = []
        gold = []

        model_to_eval = model.module if hasattr(model, 'module') else model

        with torch.no_grad(), tqdm(dev) as pbar:
            for batch in pbar:
                batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output_ids, logits = model_to_eval.generate(batch['input_ids'], depth2label=depth2label)
                for out, g in zip(output_ids, batch['labels']):
                    pred.append(set([i for i in out]))
                    gold.append([])
                    g = g.view(-1, num_class)
                    for ll in g:
                        for i, l in enumerate(ll):
                            if l == 1:
                                gold[-1].append(i)

        scores = evaluate(pred, gold, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        print('macro', macro_f1, 'micro', micro_f1)
        if args.wandb:
            swanlab.log({'val_macro': macro_f1, 'val_micro': micro_f1})
        early_stop_count += 1

        if macro_f1 > best_score_macro:
            best_score_macro = macro_f1
            save(macro_f1, best_score_macro, os.path.join(checkpoints_dir, args.name, 'checkpoint_best_macro.pt'))
            early_stop_count = 0

        if micro_f1 > best_score_micro:
            best_score_micro = micro_f1
            save(micro_f1, best_score_micro, os.path.join(checkpoints_dir, args.name, 'checkpoint_best_micro.pt'))
            early_stop_count = 0

        save(micro_f1, best_score_micro, os.path.join(checkpoints_dir, args.name, 'checkpoint_last.pt'))
        if args.wandb:
            swanlab.log({'best_macro': best_score_macro, 'best_micro': best_score_micro})

        torch.cuda.empty_cache()

    test = DataLoader(dataset['test'], batch_size=8, shuffle=False)
    model.eval()

    def test_function(extra):
        checkpoint = torch.load(os.path.join(checkpoints_dir, args.name, f'checkpoint_best{extra}.pt'),
                                map_location='cpu', weights_only=False)
        model_to_eval = model.module if hasattr(model, 'module') else model
        model_to_eval.load_state_dict(checkpoint['param'])

        pred = []
        gold = []
        with torch.no_grad(), tqdm(test) as pbar:
            for batch in pbar:
                batch = {k: v.to('cuda') for k, v in batch.items()}
                output_ids, logits = model_to_eval.generate(batch['input_ids'], depth2label=depth2label)
                for out, g in zip(output_ids, batch['labels']):
                    pred.append(set([i for i in out]))
                    gold.append([])
                    g = g.view(-1, num_class)
                    for ll in g:
                        for i, l in enumerate(ll):
                            if l == 1:
                                gold[-1].append(i)
        scores = evaluate(pred, gold, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        print('macro', macro_f1, 'micro', micro_f1)

        with open(os.path.join(checkpoints_dir, args.name, f'result{extra}.txt'), 'w', encoding='utf-8') as f:
            print('macro', macro_f1, 'micro', micro_f1, file=f)
            prefix = 'test' + extra
        if args.wandb:
            swanlab.log({prefix + '_macro': macro_f1, prefix + '_micro': micro_f1})

    test_function('_macro')
