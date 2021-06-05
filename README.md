# Deriving Word Vectors from Contextualized Language Models\\ using Topic-Aware Mention Selection

In this project, we propose a method for learning word representations that follows this basic strategy, but differs from standard word embeddings in two important ways. First, we take advantage of contextualized language models (CLMs) rather than bags of word vectors to encode contexts. Second, rather than learning a word vector directly, we use a topic model to partition the contexts in which words appear, and then learn different topic-specific vectors for each word.

This repository will guide you step to step to obtain word-topic-vectors. Also, you could skip these steps and directly download all the word-vector as tested in our paper. 

# building word-topic-vectors from scratch

## Download clean and pre-processed Wikipedia corpus

- The link to download the corpus:https://cf-my.sharepoint.com/:u:/g/personal/wangy306_cardiff_ac_uk/EYJR4aNwc0pJprgI7dh9TeIBIn5bjcsIQTrB0cyt2A1AOQ?e=0H89AS
## collect all the words from the evaluation dataset (McRae-Feature-Norm, CSLB, WordNet-SuperSense, BabelNet-Domain in our case)

- the link to download these datasets: https://cf-my.sharepoint.com/:u:/r/personal/wangy306_cardiff_ac_uk/Documents/repo.zip?csf=1&web=1&e=aUQCpl
- Run command: `python3 words_collection.py`
- The outputfile is: "all_words.txt", this text file include all the word that we are going to find mentions in the wiki-corpus. You can also use words from your interested dataset instead. 

## Apply LDA on wiki-dump:
- In this step we apply LDA (Latent Dirichlet allocation) on wiki-dump and get k-topic mixed distributions.
- Follow the instruction on this tortual (https://radimrehurek.com/gensim/wiki.html) and get required files (tf-idf and something else) for wikidump:
- run LDA on the output documents: `python3 LDA_model.py` (This command generate a json file maps each wiki-tilte to its topic distributions)
- you can also skip this section and directly download the wiki-topic-distribution from here: https://cf-my.sharepoint.com/:u:/g/personal/wangy306_cardiff_ac_uk/EQGaudFrhFdFllXBh180TEUBS_eXrGLapKex4o3sv98zog?e=kDGVKg

## Sentence sampling:
- collect sample sentences for using two strategies (C and T types respectively)
- file required: 
  all words collection: https://cf-my.sharepoint.com/:t:/g/personal/wangy306_cardiff_ac_uk/EXg5FWbRhLVDlXrPAd0vwCUBNkMTiJGiSRTFQtaYtOycaA?e=nSpwYY
  25_topic_model: https://cf-my.sharepoint.com/:u:/g/personal/wangy306_cardiff_ac_uk/EQGaudFrhFdFllXBh180TEUBS_eXrGLapKex4o3sv98zog?e=kDGVKg

- Run command `python3 WTV1.py` (the output files from above step are required)
- Run command `python3 WTV12.py`

- The output files are two documents, one containing words as foler with their selected topic-mention sentences inside, the other containing words as files with general sentences inside.

## Get vectors from sampling sentences using popular language models (BERT-BASE,BERT-LARGE,ROBERTA-BASE, ROBERTA-LARGE)
- Run command `python3 word_topic_vector.py`

# Download word Vectors

- [Mask, unmask, and average of unmask layers for McRae dataset with BERT large](https://filesender.renater.fr/?s=download&token=b3375b5e-78e6-41e0-98cb-b530d4803711)





## Get baseline embeddings(skip-gram, CBOW, glove)
- [Skip-gram trained on Wikipedia](https://cf-my.sharepoint.com/:t:/g/personal/wangy306_cardiff_ac_uk/ERPqned64qRFv-ri5_jN0CIB5z2V7XlKD9I3qm93A80wAw?e=Uu3LvF)
- [CBOW trained on Wikipedia](https://drive.google.com/file/d/171iSHR6GcL3k4IB2JsblHJuifoFarmFZ/view?usp=sharing)
- [Glove trained on Wikipedia](https://cf-my.sharepoint.com/:t:/g/personal/wangy306_cardiff_ac_uk/ESwBA0GD3mRNklhdYVQro08BuBVhLiZRDWX5Lb7uFqialw?e=XDfNnw)

## Run evaluations on lexical feature classification
- Run command `python3 word_fea_classification.py -language-model -data-set`

