# Deriving Word Vectors from Contextualized Language Models - using Topic-Aware Mention Selection

In this project, we propose a method for learning word representations that follows this basic strategy, but differs from standard word embeddings in two important ways. First, we take advantage of contextualized language models (CLMs) rather than bags of word vectors to encode contexts. Second, rather than learning a word vector directly, we use a topic model to partition the contexts in which words appear, and then learn different topic-specific vectors for each word.

This repository will guide you step to step to obtain word-topic-vectors. Also, you could skip these steps and directly download all the word-vector as tested in our paper. 

# building word-topic-vectors from scratch

## collect all the words from the evaluation dataset (McRae-Feature-Norm, CSLB, WordNet-SuperSense, BabelNet-Domain in our case)

- the link to download these words: https://cf-my.sharepoint.com/:t:/g/personal/wangy306_cardiff_ac_uk/EXg5FWbRhLVDlXrPAd0vwCUBNkMTiJGiSRTFQtaYtOycaA?e=nSpwYY
- You can also prepare a txt file containing all words from your interested dataset instead. 

## Apply LDA on wiki-dump:
- In this step we apply LDA (Latent Dirichlet allocation) on wiki-dump and get k-topic mixed distributions.
- Follow the beginning part of instruction on this tortual (https://radimrehurek.com/gensim/wiki.html) and get required files (tfidf, wordids, and cow as required for the below command line) for wikidump:
- run LDA on the output documents: `python3 LDA_model.py -k -alpha -wordids -tfidf -bow -workers -build_dir` (This command generate a json file maps each wiki-tilte to its topic distributions)
- you can also skip this section and directly download the wiki-topic-distribution in our experiment setting (k=25, alpha = 0.0001): https://cf-my.sharepoint.com/:u:/g/personal/wangy306_cardiff_ac_uk/EQGaudFrhFdFllXBh180TEUBS_eXrGLapKex4o3sv98zog?e=kDGVKg

## Sampling sentences for word vectors:
- sampling sentences using two strategies (C and T types respectively)
- file required: 
  1. a txt file containing all the words to have vectors
  2. processed-wiki-corpus:https://cf-my.sharepoint.com/:u:/g/personal/wangy306_cardiff_ac_uk/EYJR4aNwc0pJprgI7dh9TeIBIn5bjcsIQTrB0cyt2A1AOQ?e=0H89AS
  3. wiki-topic-distribution: (following our experiment setting, we choose k=25 as number of topics. This file can be downloaded from above section)

- Run command `python3 word_wiki_mention.py`
  (this line generate a nested dictionary in which each word maps to wikipages that mention it which is assocaiated with the number of that word's occurance)
- Run command `python3 sampling_sentences.py` 
- Until now, sentence sampling is completed, there are two output folders, one containing words as foler with their selected topic-mention sentences inside (strategy for T-type), the other containing words as files with general sentences inside (strategy for C-type).

## Get vectors from sampling sentences using contextual language models (BERT<base,large>,ROBERTA<base,large>)
- Run command `python3 word_topic_vector.py` (this script is missing... upload it soon)

# Download word Vectors

- [Mask, unmask, and average of unmask layers for McRae dataset with BERT large](https://filesender.renater.fr/?s=download&token=b3375b5e-78e6-41e0-98cb-b530d4803711)





## Get baseline embeddings(skip-gram, CBOW, glove)
- [Skip-gram trained on Wikipedia](https://cf-my.sharepoint.com/:t:/g/personal/wangy306_cardiff_ac_uk/ERPqned64qRFv-ri5_jN0CIB5z2V7XlKD9I3qm93A80wAw?e=Uu3LvF)
- [CBOW trained on Wikipedia](https://drive.google.com/file/d/171iSHR6GcL3k4IB2JsblHJuifoFarmFZ/view?usp=sharing)
- [Glove trained on Wikipedia](https://cf-my.sharepoint.com/:t:/g/personal/wangy306_cardiff_ac_uk/ESwBA0GD3mRNklhdYVQro08BuBVhLiZRDWX5Lb7uFqialw?e=XDfNnw)

## Run evaluations on lexical feature classification
- Run command `python3 word_fea_classification.py -language-model -data-set`
