# Deriving Word Vectors from Contextualized Language Models - using Topic-Aware Mention Selection

## Introduction

Static word vectors remain important in applications where word meaning has to be modelled in the absence of context. 

In our paper (Deriving Word Vectors from Contextualized Language Models - using Topic-Aware Mention Selection), we propose a method for learning high-quality static word vector by taking advantage of contextualized language models (BERT,RoBERTa). Instread of representing a word as a single vector, we model each word as several vectors which derive its semantic properties from articles of different topics (economics, politics, education, and so on).

For example, our representation for word "banana" have several topics-specific vectors corresponding to the topics of food, biology, industury. This make our static word vector become context sensitive in some sense. Not only does these topic-specifc-vector take advantage of diversived context, the topic partition    

This open source implementaton will guide you step by step to obtain these word-topic-vectors. Also, you could skip these steps and directly download all the word-vector experimented in our paper. 

## Building word-topic-vectors from scratch

### collect all the words from the evaluation dataset (McRae-Feature-Norm, CSLB, WordNet-SuperSense, BabelNet-Domain in our case)

- the link to download these words: https://cf-my.sharepoint.com/:t:/g/personal/wangy306_cardiff_ac_uk/EXg5FWbRhLVDlXrPAd0vwCUBNkMTiJGiSRTFQtaYtOycaA?e=nSpwYY
- You can also prepare a txt file containing all words from your interested dataset instead. 

### Apply LDA on wiki-dump:
- In this step we apply LDA (Latent Dirichlet allocation) on wiki-dump and get k-topic mixed distributions.
- Follow the beginning part of instruction on this tutorial (https://radimrehurek.com/gensim/wiki.html) and get required files (tfidf, wordids, and cow as required for the below command line) for wikidump
- run LDA on the output documents: `python3 LDA_model.py -k -alpha -wordids -tfidf -bow -workers -build_dir` (This command generate a json file maps each wiki-tilte to its topic distributions)
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

