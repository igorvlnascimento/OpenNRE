from . import encoder
from . import model
from . import framework
import torch
import os
import sys
import json
import numpy as np
import logging

root_url = "https://thunlp.oss-cn-qingdao.aliyuncs.com/"
default_root_path = os.path.join(os.getenv('HOME'), '.opennre')

def check_root(root_path=default_root_path):
    if not os.path.exists(root_path):
        os.mkdir(root_path)
        os.mkdir(os.path.join(root_path, 'benchmark'))
        os.mkdir(os.path.join(root_path, 'pretrain'))
        os.mkdir(os.path.join(root_path, 'pretrain/nre'))

def download_wiki80(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/wiki80')):
        os.mkdir(os.path.join(root_path, 'benchmark/wiki80'))
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/wiki80') + ' ' + root_url + 'opennre/benchmark/wiki80/wiki80_rel2id.json')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/wiki80') + ' ' + root_url + 'opennre/benchmark/wiki80/wiki80_train.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/wiki80') + ' ' + root_url + 'opennre/benchmark/wiki80/wiki80_val.txt')

def download_tacred(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/tacred')):
        os.mkdir(os.path.join(root_path, 'benchmark/tacred'))
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/tacred') + ' ' + root_url + 'opennre/benchmark/tacred/tacred_rel2id.json')
        logging.info('Due to copyright limits, we only provide rel2id for TACRED. Please download TACRED manually and convert the data to OpenNRE format if needed.')

def download_nyt10(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/nyt10')):
        os.mkdir(os.path.join(root_path, 'benchmark/nyt10'))
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/nyt10') + ' ' + root_url + 'opennre/benchmark/nyt10/nyt10_rel2id.json')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/nyt10') + ' ' + root_url + 'opennre/benchmark/nyt10/nyt10_train.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/nyt10') + ' ' + root_url + 'opennre/benchmark/nyt10/nyt10_test.txt')

def download_nyt10m(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/nyt10m')):
        os.mkdir(os.path.join(root_path, 'benchmark/nyt10m'))
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/nyt10m') + ' ' + root_url + 'opennre/benchmark/nyt10m/nyt10m_rel2id.json')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/nyt10m') + ' ' + root_url + 'opennre/benchmark/nyt10m/nyt10m_train.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/nyt10m') + ' ' + root_url + 'opennre/benchmark/nyt10m/nyt10m_test.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/nyt10m') + ' ' + root_url + 'opennre/benchmark/nyt10m/nyt10m_val.txt')

def download_wiki20m(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/wiki20m')):
        os.mkdir(os.path.join(root_path, 'benchmark/wiki20m'))
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/wiki20m') + ' ' + root_url + 'opennre/benchmark/wiki20m/wiki20m_rel2id.json')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/wiki20m') + ' ' + root_url + 'opennre/benchmark/wiki20m/wiki20m_train.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/wiki20m') + ' ' + root_url + 'opennre/benchmark/wiki20m/wiki20m_test.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/wiki20m') + ' ' + root_url + 'opennre/benchmark/wiki20m/wiki20m_val.txt')

def download_wiki_distant(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/wiki_distant')):
        os.mkdir(os.path.join(root_path, 'benchmark/wiki_distant'))
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/wiki_distant') + ' ' + root_url + 'opennre/benchmark/wiki_distant/wiki_distant_rel2id.json')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/wiki_distant') + ' ' + root_url + 'opennre/benchmark/wiki_distant/wiki_distant_train.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/wiki_distant') + ' ' + root_url + 'opennre/benchmark/wiki_distant/wiki_distant_test.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/wiki_distant') + ' ' + root_url + 'opennre/benchmark/wiki_distant/wiki_distant_val.txt')

def download_semeval(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/semeval')):
        os.mkdir(os.path.join(root_path, 'benchmark/semeval'))
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/semeval') + ' ' + root_url + 'opennre/benchmark/semeval/semeval_rel2id.json')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/semeval') + ' ' + root_url + 'opennre/benchmark/semeval/semeval_train.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/semeval') + ' ' + root_url + 'opennre/benchmark/semeval/semeval_test.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/semeval') + ' ' + root_url + 'opennre/benchmark/semeval/semeval_val.txt')

def download_semeval2018(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/semeval2018')):
        os.mkdir(os.path.join(root_path, 'benchmark/semeval2018'))
        os.system('wget --no-check-certificate -P ' + os.path.join(root_path, 'benchmark/semeval2018') + ' ' + "'https://docs.google.com/uc?export=download&id=1jNnqR7HeY4w6iCPqY5Z1so9hx2IYMZqK' -O benchmark/semeval2018/semeval2018_rel2id.json")
        os.system('wget --no-check-certificate -P ' + os.path.join(root_path, 'benchmark/semeval2018') + ' ' + "'https://docs.google.com/uc?export=download&id=1Deg38sUJ13J55GWlNG67SUN8ZotSg9j9' -O benchmark/semeval2018/semeval2018_train.txt")
        os.system('wget --no-check-certificate -P ' + os.path.join(root_path, 'benchmark/semeval2018') + ' ' + "'https://docs.google.com/uc?export=download&id=1k0VBMq-VPD7_a2endtxXlGT2j1TSSM72' -O benchmark/semeval2018/semeval2018_test.txt")
        os.system('wget --no-check-certificate -P ' + os.path.join(root_path, 'benchmark/semeval2018') + ' ' + "'https://docs.google.com/uc?export=download&id=1XBWx_UzGfekVuwt5MYdaTNNyvzRVshb6' -O benchmark/semeval2018/semeval2018_val.txt")
        os.system('wget --no-check-certificate -P ' + os.path.join(root_path, 'benchmark/semeval2018') + ' ' + "'https://docs.google.com/uc?export=download&id=1kycnRnvRhjC4y-_B9_rJUZrNnGzPw8br' -O benchmark/semeval2018/semeval2018_train_gpt.txt")
        # os.system('wget --no-check-certificate -P ' + os.path.join(root_path, 'benchmark/semeval2018') + ' ' + "'https://docs.google.com/uc?export=download&id=1lowaa1xW8i96-CjisCPyX3jaug8aZlmC' -O benchmark/semeval2018/semeval2018_synt_train.txt")
        # os.system('wget --no-check-certificate -P ' + os.path.join(root_path, 'benchmark/semeval2018') + ' ' + "'https://docs.google.com/uc?export=download&id=19vCZy7142xL7ODHChOno_RzRvcfeo03H' -O benchmark/semeval2018/semeval2018_synt_val.txt")

def download_ddi(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/ddi')):
        os.mkdir(os.path.join(root_path, 'benchmark/ddi'))
        os.system('wget --no-check-certificate -P ' + os.path.join(root_path, 'benchmark/ddi') + ' ' + "'https://docs.google.com/uc?export=download&id=1YUVHGLSOk-ySbhV3HqySa3RFVcB8Xxti' -O benchmark/ddi/ddi_rel2id.json")
        os.system('wget --no-check-certificate -P ' + os.path.join(root_path, 'benchmark/ddi') + ' ' + "'https://docs.google.com/uc?export=download&id=1QbpwcJbDf2TPbWf28kr7tmNPZMli5z2w' -O benchmark/ddi/ddi_train.txt")
        os.system('wget --no-check-certificate -P ' + os.path.join(root_path, 'benchmark/ddi') + ' ' + "'https://docs.google.com/uc?export=download&id=1oIIBOtTdNe8sq54txe6hh9wARQ7AwC6n' -O benchmark/ddi/ddi_test.txt")
        os.system('wget --no-check-certificate -P ' + os.path.join(root_path, 'benchmark/ddi') + ' ' + "'https://docs.google.com/uc?export=download&id=1MrXUcs_4RcOHzjtumI8lO3tUUYnAS6rw' -O benchmark/ddi/ddi_val.txt")
        os.system('wget --no-check-certificate -P ' + os.path.join(root_path, 'benchmark/ddi') + ' ' + "'https://docs.google.com/uc?export=download&id=1ECw-ub-1kdxD9mV1IIl6V6L6wz-TjHa2' -O benchmark/ddi/ddi_train_gpt.txt")
        os.system('wget --no-check-certificate -P ' + os.path.join(root_path, 'benchmark/ddi') + ' ' + "'https://docs.google.com/uc?export=download&id=187_7STdzdeyxTIEd-tgAynqvgGsPiuVP' -O benchmark/ddi/ddi_synt_train.txt")
        os.system('wget --no-check-certificate -P ' + os.path.join(root_path, 'benchmark/ddi') + ' ' + "'https://docs.google.com/uc?export=download&id=1VfzaAe9sgfUt_RS-GvhVvb1vfh87cWDE' -O benchmark/ddi/ddi_synt_val.txt")
        os.system('wget --no-check-certificate -P ' + os.path.join(root_path, 'benchmark/ddi') + ' ' + "'https://docs.google.com/uc?export=download&id=1cU8fbgSKhiNF-6k3WDG5n4OtcauFIhCM' -O benchmark/ddi/ddi_all_synt.txt")

def download_glove(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'pretrain/glove')):
        os.mkdir(os.path.join(root_path, 'pretrain/glove'))
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/glove') +  ' ' + root_url + 'opennre/pretrain/glove/glove.6B.50d_mat.npy')
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/glove') +  ' ' + root_url + 'opennre/pretrain/glove/glove.6B.50d_word2id.json')

def download_bert_base_uncased(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'pretrain/bert-base-uncased')):
        os.mkdir(os.path.join(root_path, 'pretrain/bert-base-uncased'))
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/bert-base-uncased') + ' ' + root_url + 'opennre/pretrain/bert-base-uncased/config.json')
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/bert-base-uncased') + ' ' + root_url + 'opennre/pretrain/bert-base-uncased/pytorch_model.bin')
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/bert-base-uncased') + ' ' + root_url + 'opennre/pretrain/bert-base-uncased/vocab.txt')

def download_pretrain(model_name, root_path=default_root_path):
    ckpt = os.path.join(root_path, 'pretrain/nre/' + model_name + '.pth.tar')
    if not os.path.exists(ckpt):
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/nre')  + ' ' + root_url + 'opennre/pretrain/nre/' + model_name + '.pth.tar')

def download_custom_pretrain(model_name, root_path=default_root_path):
    ckpt = os.path.join(root_path, 'pretrain/nre/' + model_name + '.pth.tar')
    print("ckpt:",ckpt)
    if not os.path.exists(ckpt):
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/nre')  + ' ' + f"'https://docs.google.com/uc?export=download&id=17bmMy2njN28JKtFcYpteMU_6vD270joD' -O {root_path}/pretrain/nre/{model_name}.pth.tar")

def download(name, root_path=default_root_path):
    if not os.path.exists(os.path.join(root_path, 'benchmark')):
        os.mkdir(os.path.join(root_path, 'benchmark'))
    if not os.path.exists(os.path.join(root_path, 'pretrain')):
        os.mkdir(os.path.join(root_path, 'pretrain'))
    if name == 'nyt10':
        download_nyt10(root_path=root_path)
    elif name == 'nyt10m':
        download_nyt10m(root_path=root_path)
    elif name == 'wiki20m':
        download_wiki20m(root_path=root_path)
    elif name == 'wiki_distant':
        download_wiki_distant(root_path=root_path)
    elif name == 'semeval':
        download_semeval(root_path=root_path)
    elif name == 'semeval2018':
        download_semeval2018(root_path=root_path)
    elif name == 'wiki80':
        download_wiki80(root_path=root_path)
    elif name == 'tacred':
        download_tacred(root_path=root_path)
    elif name == 'ddi':
        download_ddi(root_path=root_path)
    elif name == 'glove':
        download_glove(root_path=root_path)
    elif name == 'bert_base_uncased':
        download_bert_base_uncased(root_path=root_path)
    else:
        raise Exception('Cannot find corresponding data.')

def get_model(model_name, root_path=default_root_path):
    check_root()
    ckpt = os.path.join(root_path, './pretrain/nre/' + model_name + '.pth.tar')
    if model_name == 'wiki80_cnn_softmax':
        download_pretrain(model_name, root_path=root_path)
        download('glove', root_path=root_path)
        download('wiki80', root_path=root_path)
        wordi2d = json.load(open(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_word2id.json')))
        word2vec = np.load(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_mat.npy'))
        rel2id = json.load(open(os.path.join(root_path, 'benchmark/wiki80/wiki80_rel2id.json')))
        sentence_encoder = encoder.CNNEncoder(token2id=wordi2d,
                                                     max_length=40,
                                                     word_size=50,
                                                     position_size=5,
                                                     hidden_size=230,
                                                     blank_padding=True,
                                                     kernel_size=3,
                                                     padding_size=1,
                                                     word2vec=word2vec,
                                                     dropout=0.5)
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        return m
    elif model_name in ['wiki80_bert_softmax', 'wiki80_bertentity_softmax']:
        download_pretrain(model_name, root_path=root_path)
        download('bert_base_uncased', root_path=root_path)
        download('wiki80', root_path=root_path)
        rel2id = json.load(open(os.path.join(root_path, 'benchmark/wiki80/wiki80_rel2id.json')))
        if 'entity' in model_name:
            sentence_encoder = encoder.BERTEntityEncoder(
                max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased'))
        else:
            sentence_encoder = encoder.BERTEncoder(
                max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased'))
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        return m
    elif model_name in ['tacred_bert_softmax', 'tacred_bertentity_softmax']:
        download_pretrain(model_name, root_path=root_path)
        download('bert_base_uncased', root_path=root_path)
        download('tacred', root_path=root_path)
        rel2id = json.load(open(os.path.join(root_path, 'benchmark/tacred/tacred_rel2id.json')))
        if 'entity' in model_name:
            sentence_encoder = encoder.BERTEntityEncoder(
                max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased'))
        else:
            sentence_encoder = encoder.BERTEncoder(
                max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased'))
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        return m
    elif 'bert' in model_name:
        download_custom_pretrain(model_name, root_path=root_path)
        download('bert_base_uncased', root_path=root_path)
        download('ddi', root_path=root_path)
        rel2id = json.load(open(os.path.join('.', 'benchmark/ddi/ddi_rel2id.json')))
        if 'entity' in model_name:
            sentence_encoder = encoder.BERTEntityEncoder(
                max_length=128, pretrain_path=os.path.join('./opennre', 'pretrain/bert-base-uncased'))
        else:
            sentence_encoder = encoder.BERTEncoder(
                max_length=128, pretrain_path=os.path.join('./opennre', 'pretrain/bert-base-uncased'))
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        return m
    else:
        raise NotImplementedError
