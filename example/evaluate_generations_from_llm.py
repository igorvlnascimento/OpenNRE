from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np

import torch
import opennre
import argparse
import evaluate

from ast import literal_eval
from tqdm import tqdm

parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--dataset', default="none", type=str,
        help='Dataset')
parser.add_argument('--llm', default="none", type=str,
        help='LLM')
parser.add_argument('--synthetic', action='store_true', 
        help='Use synthetic data')
parser.add_argument('--synthetic_rl', action='store_true', 
        help='Use RL synthetic data')

args = parser.parse_args()

bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')

root_path = '.'
if args.dataset != 'none':
    opennre.download(args.dataset, root_path=root_path)
    
train_dataset = f'{args.dataset}_all_train.txt'
if args.synthetic:
    train_dataset = f'{args.dataset}_all_synt.txt'
elif args.synthetic_rl:
    train_dataset = f'{args.dataset}_all_synt_rl.txt'

with open(f'benchmark/{args.dataset}/{train_dataset}', 'r') as gpt_txt:
    tokens_train_sentences = gpt_txt.read().splitlines()

train_sentences = []
for i, _ in enumerate(tokens_train_sentences):
  train_sentences.append(
      {'token': " ".join(literal_eval(tokens_train_sentences[i])['token']),
       'relation': literal_eval(tokens_train_sentences[i])['relation']}
  )

with open(f'benchmark/{args.dataset}/{args.dataset}_test.txt', 'r') as gpt_txt:
    tokens_test_sentences = gpt_txt.read().splitlines()

test_sentences = []
sentences_by_relation = {}
for i, _ in enumerate(tokens_test_sentences):
    sentence = " ".join(literal_eval(tokens_test_sentences[i])['token'])
    relation = literal_eval(tokens_test_sentences[i])['relation']
    if relation in sentences_by_relation:
        sentences_by_relation[relation].append(sentence)
    else:
        sentences_by_relation[relation] = [sentence]
    test_sentences.append(
       {
            'token': sentence, 
            'relation': relation
       }
    )
   
ppls = []
predictions = []
references = []
model_name = "igorvln/dare_{}_{}_byrelation_finetuning"
for train_sentence in tqdm(train_sentences):
    sentence = train_sentence['token']
    relation = train_sentence['relation']
    predictions.append(sentence)
    references.append(sentences_by_relation[relation])
    llm_model = AutoModelForCausalLM.from_pretrained(model_name.format(args.llm, args.dataset))
    llm_tokenizer = AutoTokenizer.from_pretrained(model_name.format(args.llm, args.dataset))
    inputs_wiki_text = llm_tokenizer(sentence, return_tensors = "pt")
    loss = llm_model(input_ids = inputs_wiki_text["input_ids"], labels = inputs_wiki_text["input_ids"]).loss
    ppl = torch.exp(loss)
    ppls.append(ppl.item())

def calculate_inception_score(texts, n_split=10, eps=1E-16):
    # load model
    model = opennre.get_model(f'{args.dataset}_bert-base-uncased_entity', '.')
    yhat = None
    for text in texts:
        text = literal_eval(text)
        result = model.infer({'text': " ".join(text['token']), 'h': text['h'], 't': text['t']})
        if yhat is not None:
            yhat = torch.vstack((yhat, result[2]))
        else:
            yhat = result[2]
    # enumerate splits of texts/predictions
    scores = list()
    n_part = np.floor(yhat.shape[0] / n_split)
    for i in range(n_split):
        # retrieve p(y|x)
        ix_start, ix_end = int(i * n_part), int(i * n_part + n_part)
        p_yx = yhat[ix_start:ix_end]
        # calculate p(y)
        p_yx = p_yx.detach().numpy()
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = np.mean(sum_kl_d)
        # undo the log
        is_score = np.exp(avg_kl_d)
        # store
        scores.append(is_score)
        # average across images
    is_avg, is_std = np.mean(scores), np.std(scores)
    return is_avg, is_std

is_avg, is_std = calculate_inception_score(tokens_train_sentences)

print(f"Perplexity mean: {np.mean(ppls)}")
print(f"Inception score average: {is_avg}, Inception score Std: {is_std}")
print(f"BLEU results: {bleu.compute(predictions=predictions, references=references, max_order = 4)}")
print(f"ROUGE results: {rouge.compute(predictions=predictions, references=references)}")