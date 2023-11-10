import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.lm import Vocabulary

from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np

import torch
import opennre
import argparse

from ast import literal_eval

parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--dataset', default="ddi", type=str,
        help='Dataset')
parser.add_argument('--llm', default="gpt2", type=str,
        help='LLM')
parser.add_argument('--synthetic', action='store_true', 
        help='Use synthetic data')
parser.add_argument('--synthetic_rl', action='store_true', 
        help='Use RL synthetic data')

args = parser.parse_args()

train_dataset = f'{args.dataset}_all_train.txt'
if args.synthetic:
    train_dataset = f'{args.dataset}_all_synt.txt'
elif args.synthetic_rl:
    train_dataset = f'{args.dataset}_all_synt_rl.txt'

with open(f'benchmark/{args.dataset}/{args.dataset}_all_synt.txt', 'r') as gpt_txt:
    tokens_train_sentences = gpt_txt.read().splitlines()

train_sentences = []
for i, _ in enumerate(tokens_train_sentences):
  train_sentences.append(" ".join(literal_eval(tokens_train_sentences[i])['token']))

with open(f'benchmark/{args.dataset}/{args.dataset}_test.txt', 'r') as gpt_txt:
    tokens_test_sentences = gpt_txt.read().splitlines()

test_sentences = []
for i, _ in enumerate(tokens_test_sentences):
   test_sentences.append(
       {'token': " ".join(literal_eval(tokens_test_sentences[i])['token']), 
        'relation': literal_eval(tokens_test_sentences[i])['relation']}
    )


ppls = []
model_name = "igorvln/dare_{}_{}_train_{}_finetuning"
for test_sentence in test_sentences:
    sentence = test_sentence['token']
    relation = test_sentence['relation']
    llm_model = AutoModelForCausalLM.from_pretrained(model_name.format(args.llm, args.dataset, relation))
    llm_tokenizer = AutoTokenizer.from_pretrained(model_name.format(args.llm, args.dataset, relation))
    inputs_wiki_text = llm_tokenizer(sentence, return_tensors = "pt")
    loss = llm_model(input_ids = inputs_wiki_text["input_ids"], labels = inputs_wiki_text["input_ids"]).loss
    ppl = torch.exp(loss)
    ppls.append(ppl.item())

# tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in train_sentences]

# n = 2
# train_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
# words = [word for sent in tokenized_text for word in sent]
# words.extend(["<s>", "</s>"])
# padded_vocab = Vocabulary(words)
# model = MLE(n)
# model.fit(train_data, padded_vocab)

# tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in test_sentences]

# test_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
# for test in test_data:
#    print ("MLE Estimates:", [((ngram[-1], ngram[:-1]),model.score(ngram[-1], ngram[:-1])) for ngram in test])

# test_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
# perplexities = []
# for i, test in enumerate(test_data):
#   perplexity = model.perplexity(test)
#   perplexities.append(perplexity)
  #print("PP({0}):{1}".format(test_sentences[i], perplexity))

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