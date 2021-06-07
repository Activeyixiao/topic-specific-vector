import os
import logging
import gensim
import pickle
import json
import getopt
import sys
import pandas as pd
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet as wn
import random
import numpy as np
import nltk


def init_logging_path(log_path, task_name, file_name):
    print( os.path.join(log_path,f"{task_name}/{file_name}/"))
    dir_log  = os.path.join(log_path,f"{task_name}/{file_name}/")
    if os.path.exists(dir_log) and os.listdir(dir_log):
        dir_log += f'{file_name}_{len(os.listdir(dir_log))}.log'
        with open(dir_log, 'w'):
             os.utime(dir_log, None)
    if not os.path.exists(dir_log):
        os.makedirs(dir_log)
        dir_log += f'{file_name}_{len(os.listdir(dir_log))}.log'
        with open(dir_log, 'w'):
             os.utime(dir_log, None)
    return dir_log


def main(argv):
    log_dir = os.path.join(os.getcwd(), "log/")
    logfile = ''

    title_distribution_mapping = ""
    all_words = ""
    wiki_dir = ""
    output_dir= ""

    try: 
        opts, args = getopt.getopt(argv, "hl:a:b:c:o:", ["lfile=", "title_distribution_mapping=", "all_words=", "wiki_dir=", "odir="])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('wtv1.py  -l <logfile> -a <"title_distribution_mapping> -b <all_words> -c <wiki_dir> -o <output_dir>')
            sys.exit()
        elif opt in ("-l", "--lfile"):
            logfile = arg
        elif opt in ("-a", "--title_distribution_mapping"):  # path of sentence files
            title_distribution_mapping = arg
        elif opt in ("-b", "--all_words"):
            all_words = arg    
        elif opt in ("-c", "--wiki_dir"):
            wiki_dir = arg
        elif opt in ("-o", "--odir"):  # path to store mention vector files
            output_dir = arg



    log_file_path = init_logging_path(log_dir,"mv", logfile)
    logger = logging.getLogger('server_logger')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


    #************************************************

    nltk.download('averaged_perceptron_tagger')
    lemmatizer = WordNetLemmatizer()
    f = open(title_distribution_mapping, 'rb')
    title_distribution_mapping = pickle.load(f)
    f.close()

    """in this dictionary, each word maps to its wikititle which in turn maps to its occurance"""
    """loads all the words collected from the above datasets"""
    logger.info("loads all the words collected from the above datasets")
    concept_set = set()
    with open(all_words,'r') as inf:
        for line in inf:
            word = line.strip().lower()
            concept_set.add(word)

    """in this dictionary, each word maps to its wikititle which in turn maps to its occurance"""
    logger.info("wikipedia dump")
    lemmatizer = WordNetLemmatizer()
    title_address_mapping = defaultdict(str)
    word_pages_mention = defaultdict(lambda: defaultdict(int))
    file_count=0
    all_dir = [os.path.join(wiki_dir,d) for d in os.listdir(wiki_dir)]
    for d in all_dir:
        all_file = [os.path.join(d,f) for f in os.listdir(d)]
        for file in all_file:
            with open(file,'r') as inf:
                wiki_title = inf.readline().strip()
                title_address_mapping[wiki_title]=file
                for line in inf.readlines():
                    for sentence in sent_tokenize(line.strip()):
                        word_set = set([lemmatizer.lemmatize(word) for word in word_tokenize(sentence)])
                        for word in word_set:
                            if word in concept_set:
                                word_pages_mention[word][wiki_title]+=1
            file_count += 1
            if file_count%100000==0:
                print('processed '+ str(file_count)+' files')
                logger.info('processed '+ str(file_count)+' files')

    with open(output_dir+'/wiki_data/word_pages_mention.pkl', 'w') as f:
        pickle.dump(word_pages_mention,f)
        
    with open(output_dir+'/wiki_data/title_address_mapping.pkl', 'w') as f:
        pickle.dump(title_address_mapping,f)                
                
    with open(output_dir+'/wiki_data/word_pages_mention.json', 'w') as f:
        json.dump(word_pages_mention,f)
        
    with open(output_dir+'/wiki_data/title_address_mapping.json', 'w') as f:
        json.dump(title_address_mapping,f)    

    print('wididata done')
    logger.info('wididata done')

    #********************************************************

if __name__ == '__main__':
    main(sys.argv[1:])                                    
