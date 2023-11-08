import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.lm import Vocabulary

import numpy as np

import torch
import opennre
import argparse

from ast import literal_eval

parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--dataset', default="ddi", type=str,
        help='Dataset')

args = parser.parse_args()

with open(f'benchmark/{args.dataset}/{args.dataset}_all_synt.txt', 'r') as gpt_txt:
    tokens_train_sentences = gpt_txt.read().splitlines()

train_sentences = []
for i, _ in enumerate(tokens_train_sentences):
   train_sentences.append(" ".join(literal_eval(tokens_train_sentences[i])['token']))

with open(f'benchmark/{args.dataset}/{args.dataset}_test.txt', 'r') as gpt_txt:
    tokens_test_sentences = gpt_txt.read().splitlines()

test_sentences = []
for i, _ in enumerate(tokens_test_sentences):
   test_sentences.append(" ".join(literal_eval(tokens_test_sentences[i])['token']))

tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in train_sentences]

n = 2
train_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
words = [word for sent in tokenized_text for word in sent]
words.extend(["<s>", "</s>"])
padded_vocab = Vocabulary(words)
model = MLE(n)
model.fit(train_data, padded_vocab)

tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in test_sentences]

test_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
for test in test_data:
   print ("MLE Estimates:", [((ngram[-1], ngram[:-1]),model.score(ngram[-1], ngram[:-1])) for ngram in test])

test_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
perplexities = []
for i, test in enumerate(test_data):
  perplexity = model.perplexity(test)
  perplexities.append(perplexity)
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

print(f"Median perplexity: {np.median(perplexities)}")
print(f"Inception score average: {is_avg}, Inception score Std: {is_std}")