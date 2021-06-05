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
    word_pages_mention_path = ""
    title_address_mapping_path = ""
    output_dir= ""

    try: 
        opts, args = getopt.getopt(argv, "hl:a:b:c:o:", ["lfile=", "title_distribution_mapping=", "word_pages_mention_path=", "title_address_mapping_path=", "odir="])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('wtv2.py  -l <logfile> -a <"title_distribution_mapping> -b <"word_pages_mention_path> -c <title_address_mapping_path> -o <output_dir>')
            sys.exit()
        elif opt in ("-l", "--lfile"):
            logfile = arg
        elif opt in ("-a", "--title_distribution_mapping"):  
            title_distribution_mapping = arg    
        elif opt in ("-b", "--word_pages_mention_path"):  
            word_pages_mention_path = arg
        elif opt in ("-c", "--title_address_mapping_path"):
            title_address_mapping_path = arg    
        elif opt in ("-o", "--odir"):  
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

    word_pages_mention = json.load(open(word_pages_mention_path, "rb"))
    title_address_mapping = json.load(open(title_address_mapping_path, "rb"))

    print('wididata done')
    logger.info('wididata done')

    #************************************************

    """map word to its topics and its mention pages"""
    print('word_topic_page ....')
    logger.info('word_topic_page ....')

    word_topic_page = defaultdict(lambda: defaultdict(list))
    for word in word_pages_mention.keys():
        ps = word_pages_mention[word]
        for page,times in ps.items():
            topics = title_distribution_mapping[page]
            for topic_score in topics:
                topic,score = topic_score
                if score >= 0.15:
                    score = float(round(score,3))
                    word_topic_page[word][topic].append((page,score,times))
    for word,t_p in word_topic_page.items():
        for topic in t_p.keys():
            t_p[topic] = sorted(t_p[topic],key=lambda x:x[1], reverse=True)
    for word in word_topic_page.keys():
        word_topic_page[word] = {k: v for k, v in sorted(word_topic_page[word].items(),key=lambda item:len(item[1]),reverse=True)}
            
    with open(output_dir+'/word_topic/word_topic_page.json', 'w') as f:
        json.dump(word_topic_page,f)    
 
    print('word_topic_page done')
    logger.info('word_topic_page  done')


    #************************************************ 

    """whole page part: Select the top topics for each word and stable pages for each topic"""

    print('word_topic_page_selected ...')
    logger.info(' word_topic_page_selected  ....')

    word_topic_page_selected = defaultdict(lambda: defaultdict(list))

    for word in word_topic_page.keys():
        topic_selected = []
        total_num = sum([len(word_topic_page[word][topic]) for topic in word_topic_page[word].keys()])
        threshold = int(total_num*(6/10))
        count = 0
        for topic,pages in word_topic_page[word].items():
            if len(pages)>=10:
                if count < threshold:
                    count += len(pages)
                    if len(topic_selected) >= 6:
                        break
                    topic_selected.append(topic)
                else:
                    break
            else:
                pass
        if len(topic_selected)>=1:
            for topic in topic_selected:
                ps = word_topic_page[word][topic]
                for p in ps:
                    word_topic_page_selected[word][topic].append(p)


    with open(output_dir+'/word_topic/word_topic_page_selected.json', 'w') as f:
        json.dump(word_topic_page_selected,f)    

    print('word_topic_page_selected done')
    logger.info('word_topic_page_selected  done')
    
    #************************************************ 
     
    """Given a list of words, we want get all sentences that mention it and save these sentence into a text file"""
    
    print('mention sentneces ....')
    logger.info('mention sentneces ....')

    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
os.mkdir(output_dir+'/concept_mention_page_25')
for word in word_topic_page_selected.keys():
    os.mkdir(output_dir+'/concept_mention_page_25/'+str(word))
    topics = word_topic_page_selected[word].keys()
    for topic in topics:
        file_name = output_dir+'/concept_mention_page_25/'+str(word)+'/'+str(topic)+'.txt'
        with open(file_name,'w') as outf:
            s_count=0
            outf.write(str(word)+' '+str(topic)+'\n')
            titles = [t[0] for t in word_topic_page_selected[word][topic]]
            for i in range(len(titles)):
                title = word_topic_page_selected[word][topic][i][0]
                topic_score = round(word_topic_page_selected[word][topic][i][1],3)
                #num = word_topic_page_selected[word][topic][i][2]
                working_file = title_address_mapping[title]
                if s_count < 100:
                    with open(working_file,'r') as inf:
                        #name = inf.readline().strip()
                        for line in inf.readlines():
                            if s_count < 100:
                                sentences = sent_tokenize(line.strip())
                                for s in sentences:
                                    if s_count < 100:
                                        lem_s = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(s)])
                                        word_set = set([word for word in word_tokenize(lem_s)])
                                        if len(word_set) <=60:
                                            if word in word_set:
                                                if s_count < 100:
                                                    S = str(topic_score)+'___'+lem_s
                                                    outf.write(S+'\n')
                                                    s_count += 1
                                                else:
                                                    break

    os.mkdir(output_dir+'/concept_mention_general')
    for word in word_topic_page_selected.keys():
        file_name = output_dir+'/concept_mention_general/'+str(word)+'.txt'
        with open(file_name,'w') as outf:
            s_count = 0
            titles = word_pages_mention[word].keys()
            random.shuffle(titles)
            for title in titles:
                if s_count < 500:
                    working_file = title_address_mapping[title]
                    with open(working_file,'r') as inf:
                        #name = inf.readline().strip()
                        for line in inf.readlines():
                            sentences = sent_tokenize(line.strip())
                            for s in sentences:
                                lem_s = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(s)])
                                word_set = set([word for word in word_tokenize(lem_s)])
                                if len(word_set) <=60:
                                    if word in word_set:
                                        if s_count < 500:
                                            outf.write(lem_s+'\n')
                                            s_count += 1
                                        else:
                                            break
    print('mention sentneces done')
    logger.info('mention sentneces done')                                    

if __name__ == '__main__':
    main(sys.argv[1:])                                    
