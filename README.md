# Deriving Word Vectors from Contextualized Language Models - using Topic-Aware Mention Selection

## Introduction

Static word vectors remain important in applications where word meaning has to be modelled in the absence of context. 

In our recent paper [Deriving Word Vectors from Contextualized Language Models - using Topic-Aware Mention Selection](), we propose a method for learning high-quality static word vectors by taking advantage of contextualized language models (BERT,RoBERTa). Instead of representing a word as a single vector, we model each word as several vectors, which derive its semantic properties from articles on different topics (economics, politics, education, and so on).

For example, our representation for word the word "banana" consists of several topics-specific vectors, corresponding to topics related to food, biology, industry. 

This document contains step-by-step instructions for obtaining these topic-specific word vectors. You could also skip these steps and directly download the topic-specific vectors that were used for the experiments in our paper.

## Building word-topic-vectors from scratch

### Requirements
- Python3
- Numpy
- Torch
- Gensim
- NLTK

### Step 1: Selecting a vocabulary

- The code requires a text file containing the vocabulary, i.e. the set of words for which vector representations need to be obtained. This vocabulary is encoded as a plain text file, with one word per line.

- The vocabulary corresponding to the experiments from our paper can be downloaded here: https://zenodo.org/record/4954493#.YMhkTHVKg5k

- In this case, the vocabulary consists of all words from the four evaluation dataset: the extended McRae feature norms, CSLB, WordNet supersenses and BabelNet domains

### Step 2: Applying Latent Dirichlet Allocation on a Wikipedia dump:

- Download and pre-process a Wikipedia dump, following the instructions from the first part of this tutorial ("Preparing the corpus"). At this point, you will have the required files for the following command line code: "tfidf", "wordids", and "cow".

- Run the LDA (Latent Dirichlet allocation) implementation from Gensim on the pre-processed wiki files: 

- `python3 LDA_model.py -k 25 -alpha 0.0001 -wordids wordids.txt -tfidf wiki_tiidf -bow wiki.bow -workers 1 -build_dir build_folder` 
 
- In this case, the hyper-parameter "k", which represents the number of topics, is set to 25, while the hyperparameter "alpha" is set to 0.0001. These correspond to the values that were used in our experiments. Workers is the number of CPU cores which should be used to process the data.

- The above command line code generates a JSON file which maps each Wikipedia article to its topic distribution.

- The resulting topic distribution can also be downloaded here: [wiki-topic-distribution](https://cf-my.sharepoint.com/:u:/g/personal/wangy306_cardiff_ac_uk/EQGaudFrhFdFllXBh180TEUBS_eXrGLapKex4o3sv98zog?e=kDGVKg) for our default setting (k=25, alpha = 0.0001) 

### Step 3: Sampling sentences for getting word vectors:
- For this step, the following files are needed: 
  1. A plain text file containing all the words for which a vector needs to be obtained (cf. Step 1).
  2. A preprocessed Wikipedia corpus; the corpus that we used can be found here:[wiki-file](https://cf-my.sharepoint.com/:u:/g/personal/wangy306_cardiff_ac_uk/EYJR4aNwc0pJprgI7dh9TeIBIn5bjcsIQTrB0cyt2A1AOQ?e=0H89AS)
  3. The wiki-topic-distribution JSON file obtained in Step 2.

- Run the following command:

- `python3 word_wiki_mention.py`

- This command generates a nested dictionary in which each word maps to the Wikipedia articles that mention it, which is associated with the number of occurrence of that word.
- Run the following command:
- `python3 sampling_sentences.py` 
- The above script generates two folders, both of them containing words and their selected sampling sentences. The only difference is that they make use of different strategies for sampling the sentences:
  1. folder "concept mention general": randomly select sentences for a word, from all the sentences mentioning that word 
  2. folder "concept_mention_page_25": select sample sentences for a word according to the topic distributions of that word 

### Obtaining vectors using contextual language models (BERT<base,large>,ROBERTA<base,large>)
- Run command `python3 word_topic_vector.py` (this script is missing... upload it soon)

## Download word vectors

- [all C-vectors](https://zenodo.org/record/4925042#.YMKch3VKg5l) 
- [all T-vector](https://zenodo.org/record/4921323#.YMKcvHVKg5k) 
- [all A-vector](https://zenodo.org/record/4925059#.YMKjPHVKg5k) 
- [all Tmask-pca-vector](https://zenodo.org/record/4925073#.YMKjw3VKg5k)
- [Skip-gram trained on Wikipedia](https://cf-my.sharepoint.com/:t:/g/personal/wangy306_cardiff_ac_uk/ERPqned64qRFv-ri5_jN0CIB5z2V7XlKD9I3qm93A80wAw?e=Uu3LvF)
- [CBOW trained on Wikipedia](https://drive.google.com/file/d/171iSHR6GcL3k4IB2JsblHJuifoFarmFZ/view?usp=sharing)


## Run evaluations on lexical feature classification
- Run command: `python3 nn_classifier/nnclassifier/run_network.py -dataset -vector_type -embed_path -mask -batch_size -infeatures learning_rate -weight_decay -seed -gpu -num_epoch vector_name`

