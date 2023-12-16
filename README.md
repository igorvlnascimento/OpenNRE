# OpenNRE

This repository is a subproject of THU-OpenSK, and all subprojects of THU-OpenSK are as follows.

## What is Relation Extraction

Relation extraction is a natural language processing (NLP) task aiming at extracting relations (e.g., *founder of*) between entities (e.g., **Bill Gates** and **Microsoft**). For example, from the sentence *Bill Gates founded Microsoft*, we can extract the relation triple (**Bill Gates**, *founder of*, **Microsoft**). 

Relation extraction is a crucial technique in automatic knowledge graph construction. By using relation extraction, we can accumulatively extract new relation facts and expand the knowledge graph, which, as a way for machines to understand the human world, has many downstream applications like question answering, recommender system and search engine. 

## How to Cite

A good research work is always accompanied by a thorough and faithful reference. If you use or extend our work, please cite the following paper:

```
@inproceedings{han-etal-2019-opennre,
    title = "{O}pen{NRE}: An Open and Extensible Toolkit for Neural Relation Extraction",
    author = "Han, Xu and Gao, Tianyu and Yao, Yuan and Ye, Deming and Liu, Zhiyuan and Sun, Maosong",
    booktitle = "Proceedings of EMNLP-IJCNLP: System Demonstrations",
    year = "2019",
    url = "https://www.aclweb.org/anthology/D19-3029",
    doi = "10.18653/v1/D19-3029",
    pages = "169--174"
}
```

## Install 

### Install as A Python Package

We are now working on deploy OpenNRE as a Python package. Coming soon!

### Using Git Repository

Clone the repository from our github page (don't forget to star us!)

```bash
git clone https://github.com/thunlp/OpenNRE.git
```

If it is too slow, you can try
```
git clone https://github.com/thunlp/OpenNRE.git --depth 1
```

Then install all the requirements:

```
pip install -r requirements.txt
```

**Note**: Please choose appropriate PyTorch version based on your machine (related to your CUDA version). For details, refer to https://pytorch.org/. 

Then install the package with 
```
python setup.py install 
```

If you also want to modify the code, run this:
```
python setup.py develop
```

## Training example

Fine-tuning GPT-2 to be used to the DDI 2013 dataset domain:
```bash
python example/finetuning_llm.py \
    --llm gpt2 \
    --dataset ddi \
```

Fine-tuning GPT-2 on DDI 2013 to generate sentences for each relationship given a query:
```bash
python example/finetuning_llm_per_relation.py \
    --llm gpt2 \
    --dataset ddi \
    --generator_model igorvln/dare_gpt2_ddi_finetuning
```

Fine-tuning GPT-2 optimized by reinforcement learning using [TRL|https://github.com/huggingface/trl] framework on DDI 2013:
```bash
python example/finetuning_llm_with_rl.py \
    --dataset ddi \
    --classifier ddi_bert-base-uncased_entity
```

To execute the above code you have to login on [Wandb|https://docs.wandb.ai/ref/cli/wandb-login]

To reproduce training on BERT model for DDI dataset with the entity pooler:
```bash
python example/train_n_supervised_bert.py \
    --batch_size 16
    --dataset ddi
    --pooler entity
```

To reproduce training on BERT model for DDI dataset with the cls pooler:
```bash
python example/train_n_supervised_bert.py \
    --batch_size 16
    --dataset ddi
    --pooler cls
```

To reproduce training on BERT model for DDI synthetic dataset with the entity pooler:
```bash
python example/train_n_supervised_bert.py \
    --batch_size 16
    --dataset ddi
    --pooler entity
    --synthetic
```

To reproduce training on BERT model for DDI RL synthetic dataset with the entity pooler:
```bash
python example/train_n_supervised_bert.py \
    --batch_size 16
    --dataset ddi
    --pooler entity
    --synthetic_rl
```

We provide many options in the example training code and you can check them out for detailed instructions.
