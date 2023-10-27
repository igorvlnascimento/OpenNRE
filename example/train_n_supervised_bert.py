# coding:utf-8
import torch
import numpy as np
import json
import opennre
from opennre import model, framework
import sys
import os
import argparse
import logging
import random
from pathlib import Path
from datetime import datetime

from mlxtend.plotting import plot_confusion_matrix

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', default='bert-base-uncased', 
        help='Pre-trained ckpt path / model name (hugginface)')
parser.add_argument('--ckpt', default='', 
        help='Checkpoint name')
parser.add_argument('--pooler', default='entity', choices=['cls', 'entity'], 
        help='Sentence representation pooler')
parser.add_argument('--only_test', action='store_true', 
        help='Only run test')
parser.add_argument('--mask_entity', action='store_true', 
        help='Mask entity mentions')

# Data
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
        help='Metric for picking up best checkpoint')
parser.add_argument('--dataset', default='none', choices=['none', 'semeval', 'wiki80', 'tacred', 'ddi'], 
        help='Dataset. If not none, the following args can be ignored')
parser.add_argument('--train_file', default='', type=str,
        help='Training data file')
parser.add_argument('--val_file', default='', type=str,
        help='Validation data file')
parser.add_argument('--test_file', default='', type=str,
        help='Test data file')
parser.add_argument('--rel2id_file', default='', type=str,
        help='Relation to ID file')

# Hyper-parameters
parser.add_argument('--batch_size', default=64, type=int,
        help='Batch size')
parser.add_argument('--lr', default=2e-5, type=float,
        help='Learning rate')
parser.add_argument('--max_length', default=128, type=int,
        help='Maximum sentence length')
parser.add_argument('--max_epoch', default=3, type=int,
        help='Max number of training epochs')

# Trials
parser.add_argument('--trials', default=5, type=int,
        help='Trials')

args = parser.parse_args()

SEEDS_PATH = Path('opennre')
SEEDS_TEXT_FILE = SEEDS_PATH / 'seeds.txt'

if os.path.exists(SEEDS_TEXT_FILE):
    with open(SEEDS_TEXT_FILE, 'r') as seeds_file:
        SEEDS = seeds_file.readlines()
else:
    SEEDS = [str(random.randint(10**6, 10**7 - 1)) for _ in range(args.trials)]
    with open(SEEDS_TEXT_FILE, 'w') as seeds_file:
        seeds_file.write('\n'.join(SEEDS))

# Some basic settings
root_path = '.'
sys.path.append(root_path)
CKPT_PATH = Path('ckpt')
DATASET_PATH = CKPT_PATH / args.dataset
PRETRAIN_PATH = DATASET_PATH / args.pretrain_path
POOLER_PATH = PRETRAIN_PATH /  f"{args.pooler}_mask_entity" if args.mask_entity else args.pooler
DATETIME_PATH = POOLER_PATH / datetime.now().astimezone().strftime("%Y-%m-%d_%H:%M:%S")
if args.dataset is not None:
    DATETIME_PATH.mkdir(parents=True, exist_ok=True)
if len(args.ckpt) == 0:
    args.ckpt = '{}_{}_{}'.format(args.dataset, args.pretrain_path, args.pooler)

if args.dataset != 'none':
    opennre.download(args.dataset, root_path=root_path)
    args.train_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_train.txt'.format(args.dataset))
    args.val_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_val.txt'.format(args.dataset))
    args.test_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_test.txt'.format(args.dataset))
    if not os.path.exists(args.test_file):
        logging.warn("Test file {} does not exist! Use val file instead".format(args.test_file))
        args.test_file = args.val_file
    args.rel2id_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_rel2id.json'.format(args.dataset))
    if args.dataset == 'wiki80':
        args.metric = 'acc'
    else:
        args.metric = 'micro_f1'
else:
    if not (os.path.exists(args.train_file) and os.path.exists(args.val_file) and os.path.exists(args.test_file) and os.path.exists(args.rel2id_file)):
        raise Exception('--train_file, --val_file, --test_file and --rel2id_file are not specified or files do not exist. Or specify --dataset')

logging.info('Arguments:')
for arg in vars(args):
    logging.info('    {}: {}'.format(arg, getattr(args, arg)))

rel2id = json.load(open(args.rel2id_file))

acc, micro_f1, macro_f1, weighted_f1 = [], [], [], []
for i in range(args.trials):

    # Define the sentence encoder
    if args.pooler == 'entity':
        sentence_encoder = opennre.encoder.BERTEntityEncoder(
            max_length=args.max_length, 
            pretrain_path=args.pretrain_path,
            mask_entity=args.mask_entity
        )
    elif args.pooler == 'cls':
        sentence_encoder = opennre.encoder.BERTEncoder(
            max_length=args.max_length, 
            pretrain_path=args.pretrain_path,
            mask_entity=args.mask_entity
        )
    else:
        raise NotImplementedError

    # Define the model
    model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

    ckpt = DATETIME_PATH / '{}_{}.pth.tar'.format(args.ckpt, i+1)

    # Define the whole training framework
    framework = opennre.framework.SentenceRE(
        train_path=args.train_file,
        val_path=args.val_file,
        test_path=args.test_file,
        model=model,
        ckpt=ckpt,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        lr=args.lr,
        opt='adamw'
    )

    # Set random seed
    set_seed(int(SEEDS[i]))

    # Train the model
    if not args.only_test:
        framework.train_model('micro_f1')

    # Test
    framework.load_state_dict(torch.load(ckpt)['state_dict'])
    result = framework.eval_model(framework.test_loader)

    with open(DATETIME_PATH / f'results_{i+1}.txt', 'w') as results_file:
        results_file.write(str(result))

    acc.append(result['acc'])
    micro_f1.append(result['micro_f1'])
    macro_f1.append(result['macro_f1'])
    weighted_f1.append(result['weighted_f1'])

    # Print the result
    logging.info('Test set results:')
    logging.info('Accuracy: {}'.format(result['acc']))
    logging.info('Micro precision: {}'.format(result['micro_p']))
    logging.info('Micro recall: {}'.format(result['micro_r']))
    logging.info('Micro F1: {}'.format(result['micro_f1']))
    logging.info('Macro F1: {}'.format(result['macro_f1']))
    logging.info('Weighted F1: {}'.format(result['weighted_f1']))

    fig, _ = plot_confusion_matrix(
        conf_mat=result['confusion_matrix'],
        class_names=rel2id,
        figsize=(10, 7)
    )
    fig.savefig(DATETIME_PATH / f'confusion_matrix_{i+1}.png')

with open(DATETIME_PATH / 'mean_std_results.txt', 'w') as mean_file:
    mean_file.write('Accuracy mean: {} +- {}\n'.format(np.mean(acc), np.std(acc)))
    mean_file.write('Micro F1 mean: {} +- {}\n'.format(np.mean(micro_f1), np.std(micro_f1)))
    mean_file.write('Macro F1 mean: {} +- {}\n'.format(np.mean(macro_f1), np.std(macro_f1)))
    mean_file.write('Weighted F1 mean: {} +- {}\n'.format(np.mean(weighted_f1), np.std(weighted_f1)))