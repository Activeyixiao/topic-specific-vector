# Deriving Word Vectors from Contextualized Language Models - using Topic-Aware Mention Selection

## Introduction

Static word vectors remain important in applications where word meaning has to be modelled in the absence of context. 

In our recent paper [Deriving Word Vectors from Contextualized Language Models - using Topic-Aware Mention Selection](), we propose a method for learning high-quality static word vector by taking advantage of contextualized language models (BERT,RoBERTa). Instread of representing a word as a single vector, we model each word as several vectors which derive its semantic properties from articles of different topics (economics, politics, education, and so on).

For example, our representation for word "banana" have several topics-specific vectors corresponding to the topics of food, biology, industury. This make our static word vector become context sensitive in some sense.    

This open source implementaton will guide you step by step to obtain these word-topic-vectors. You could also skip these steps and directly download all the word-vector experimented in our paper. 

## Building word-topic-vectors from scratch

### Requirements
- Python3
- Numpy
- Torch
- Gensim
- NLTK

### collect all the words from the evaluation dataset 

- Download words (words from McRae-Feature-Norm, CSLB, WordNet-SuperSense, BabelNet-Domain in our case): https://cf-my.sharepoint.com/:t:/g/personal/wangy306_cardiff_ac_uk/EXg5FWbRhLVDlXrPAd0vwCUBNkMTiJGiSRTFQtaYtOycaA?e=nSpwYY
- You could also prepare your own txt file containing all words your interested.

### Apply LDA on wiki-dump:

- Download and pre-process the wiki-article following the instruction on this [tutorial](https://radimrehurek.com/gensim/wiki.html) until you finish the first part: "Preparing the corpus". By then, you will get required large files ("tfidf", "wordids", and "cow") for the following command line code.

- run LDA (Latent Dirichlet allocation) on the pre-precessed wiki files: `python3 LDA_model.py -k 25 -alpha 0.0001 -wordids wordids.txt -tfidf wiki_tiidf -bow wiki.bow -workers 1 -build_dir build_folder` (The hyper-parameter "k" is number of topics, you can set different number on this; "alpha" is the alpha value for LDA, you can fine-tunning this value to get desired topic clusters; workers is number of cores to process, unless you want multi-processing the task to save time, set this to 1)

- The above command line code generate a json file that maps each wiki-article to its topic distributions)

- you can also skip this section and directly download the wiki-topic-distribution in our experiment setting (k=25, alpha = 0.0001): https://cf-my.sharepoint.com/:u:/g/personal/wangy306_cardiff_ac_uk/EQGaudFrhFdFllXBh180TEUBS_eXrGLapKex4o3sv98zog?e=kDGVKg

### Sampling sentences for word vectors:
- sampling sentences using two strategies (C and T types respectively)
- file required: 
  1. a txt file containing all the words to have vectors
  2. processed-wiki-corpus:https://cf-my.sharepoint.com/:u:/g/personal/wangy306_cardiff_ac_uk/EYJR4aNwc0pJprgI7dh9TeIBIn5bjcsIQTrB0cyt2A1AOQ?e=0H89AS
  3. wiki-topic-distribution: (following our experiment setting, we choose k=25 as number of topics. This file can be downloaded from above section)

- Run command `python3 word_wiki_mention.py`
  (this line generate a nested dictionary in which each word maps to wikipages that mention it which is assocaiated with the number of that word's occurance)
- Run command `python3 sampling_sentences.py` 
- Until now, sentence sampling is completed, there are two output folders, one containing words as foler with their selected topic-mention sentences inside (strategy for T-type), the other containing words as files with general sentences inside (strategy for C-type).

### Get vectors from sampling sentences using contextual language models (BERT<base,large>,ROBERTA<base,large>)
- Run command `python3 word_topic_vector.py` (this script is missing... upload it soon)

## Download word Vectors

- [all C-vectors](https://zenodo.org/record/4925042#.YMKch3VKg5l)
- [all T-vector](https://zenodo.org/record/4921323#.YMKcvHVKg5k)
- [all A-vector](https://zenodo.org/record/4925059#.YMKjPHVKg5k)
- [all Tmask-pca-vector](https://zenodo.org/record/4925073#.YMKjw3VKg5k)
- [Skip-gram trained on Wikipedia](https://cf-my.sharepoint.com/:t:/g/personal/wangy306_cardiff_ac_uk/ERPqned64qRFv-ri5_jN0CIB5z2V7XlKD9I3qm93A80wAw?e=Uu3LvF)
- [CBOW trained on Wikipedia](https://drive.google.com/file/d/171iSHR6GcL3k4IB2JsblHJuifoFarmFZ/view?usp=sharing)


## Run evaluations on lexical feature classification
- Run command: `python3 nn_classifier/nnclassifier/run_network.py -dataset -vector_type -embed_path -mask -batch_size -infeatures learning_rate -weight_decay -seed -gpu -num_epoch vector_name`

