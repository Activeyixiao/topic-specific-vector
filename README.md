This project aims at a better word vector to capture the meaning of word by using the current language model

# VECTOR STATUS

Please update [this spreadsheet](https://docs.google.com/spreadsheets/d/1uZgHYo4_bqWAHgq0w3IS9JmWBjkwY2FqxcBjEZD4btA/edit?usp=sharing)

# Steps 1: 
- get word-mention-topic-vectors
## Download clean and pre-processed Wikipedia corpus

- The link to download the corpus:https://cf-my.sharepoint.com/:u:/g/personal/wangy306_cardiff_ac_uk/EYJR4aNwc0pJprgI7dh9TeIBIn5bjcsIQTrB0cyt2A1AOQ?e=0H89AS
## collect all the words from the evaluation dataset (McRae-Feature-Norm, CSLB, WordNet-SuperSense, BabelNet-Domain, ConceptNet, BLESS, HyperLex)

- the link to download these 7 datasets: https://cf-my.sharepoint.com/:u:/r/personal/wangy306_cardiff_ac_uk/Documents/repo.zip?csf=1&web=1&e=aUQCpl
- Run command: `python3 words_collection.py`
- The outputfile is: "all_words.txt", this text file include all the word that we are going to find mentions in the wiki-corpus.

## Apply LDA on wiki-dump:
- In this step we apply LDA (Latent Dirichlet allocation) on wiki-dump and get k-topic mixed distributions.
- Follow the beginning instruction on this tortual and download the wiki-dump and get required tf-idf files:https://radimrehurek.com/gensim/wiki.html
- run LDA on the output documents: `python3 LDA_model.py`
- The result dictinary file maps each wiki-tilte to its mixed topic distributions 
- you can also skip this section and directly download the wiki-topic-distribution in here: https://cf-my.sharepoint.com/:u:/g/personal/wangy306_cardiff_ac_uk/EQGaudFrhFdFllXBh180TEUBS_eXrGLapKex4o3sv98zog?e=kDGVKg

## Sentence sampling:
- collect sample sentences for both word-topic-vector and word-general-vector
- file required: 
  all words collection: https://cf-my.sharepoint.com/:t:/g/personal/wangy306_cardiff_ac_uk/EXg5FWbRhLVDlXrPAd0vwCUBNkMTiJGiSRTFQtaYtOycaA?e=nSpwYY
  25_topic_model: https://cf-my.sharepoint.com/:u:/g/personal/wangy306_cardiff_ac_uk/EQGaudFrhFdFllXBh180TEUBS_eXrGLapKex4o3sv98zog?e=kDGVKg

- Run command `python3 WTV1.py` (the output files from above step are required)
- Run command `python3 WTV12.py`

- The output files are two documents, one containing words as foler with their selected topic-mention sentences inside, the other containing words as files with general sentences inside.

## Get vectors from sampling sentences using popular language models (BERT-BASE,BERT-LARGE,ROBERTA-BASE, ROBERTA-LARGE)
- Run command `python3 word_topic_vector.py`

### Vectors

- [Mask, unmask, and average of unmask layers for McRae dataset with BERT large](https://filesender.renater.fr/?s=download&token=b3375b5e-78e6-41e0-98cb-b530d4803711)

# Evaluation: 

Evaluate word-mention-topic-vector on three kinds of tasks:
- lexical features classification
- relation classification
- ontology completion

## Get baseline embeddings(skip-gram, CBOW, glove)
- [Skip-gram trained on Wikipedia](https://cf-my.sharepoint.com/:t:/g/personal/wangy306_cardiff_ac_uk/ERPqned64qRFv-ri5_jN0CIB5z2V7XlKD9I3qm93A80wAw?e=Uu3LvF)
- [CBOW trained on Wikipedia](https://drive.google.com/file/d/171iSHR6GcL3k4IB2JsblHJuifoFarmFZ/view?usp=sharing)
- [Glove trained on Wikipedia](https://cf-my.sharepoint.com/:t:/g/personal/wangy306_cardiff_ac_uk/ESwBA0GD3mRNklhdYVQro08BuBVhLiZRDWX5Lb7uFqialw?e=XDfNnw)

## Run evaluations on lexical feature classification
- Run command `python3 word_fea_classification.py -language-model -data-set`
## Run evaluations on relation classification
- Run command `python3 RC.py -input-data -language-model -outputfoler`
## Run ontology completion
- Run command `python3 OC.py -input-data -language-model -outputfoler`
