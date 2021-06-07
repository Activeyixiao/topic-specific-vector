import argparse
import pandas as pd
import pickle
import json
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split

def read_concepts(file):
    with open(file, 'r') as f:
        nouns = [line.strip() for line in f]
    return nouns


def word_feature_dic(evaluation_dataset_dirpath):
    if 'CSLB' in evaluation_dataset_dirpath:
        D_concept_features = defaultdict(list)
        D_index_feature = defaultdict(int)
        with open(os.path.join(evaluation_dataset_dirpath,'feature_matrix.dat'),'r') as inf:
            all_features = inf.readline().strip().split('\t')[1:]
            for INDEX,feature in enumerate(all_features):
                D_index_feature[INDEX]=feature
            all_concept_v = inf.readlines()
            for line in all_concept_v:
                line = line.strip().split('\t')
                concept = line[0]
                concept = concept.replace('_',' ')
                vector = line[1:]
                for n,v in enumerate(vector):
                    if v != str(0.0):
                        D_concept_features[concept].append(D_index_feature[n])
    
    elif 'WordNet' in evaluation_dataset_dirpath:
        with open(os.path.join(evaluation_dataset_dirpath,'WNdb-3.0/word_label.pkl'),'rb') as f:
            D_concept_features = pickle.load(f)

    elif 'BabelDomains' in evaluation_dataset_dirpath:
        D_concept_features = defaultdict(list)
        with open(os.path.join(evaluation_dataset_dirpath,'domain_vectors.txt'),'r') as inf:
            for line in inf.readlines():
                cols = line.strip().split('\t')
                domain = cols[0]
                concepts = [i.split(' ')[0] for i in cols[1:]]
                for concept in concepts:
                    D_concept_features[concept].append(domain)

    elif 'McRae' in evaluation_dataset_dirpath:
        D_concept_features = defaultdict(list)
        file = pd.read_csv(os.path.join(evaluation_dataset_dirpath,'McRae_new.csv'))
        features = list(file.columns)[1:]
        value_ls = file.values.tolist()
        for ls in value_ls:
            concept = ls[0]
            values = ls[1:]
            for inx,value in enumerate(values):
                if value!=0:
                    D_concept_features[concept].append(features[inx])

    return D_concept_features


def selected_word_featrue(D,bar,*embeddings):
    feature_word_d = defaultdict(list)
    for word in D.keys():
        found=True
        for idx,emb in enumerate(embeddings):
            if not word in emb:
                found=False
                #print(word,' not found in embedding at position ',idx)
                break
        if found:
            values = D[word]
            for value in values:
                feature_word_d[value].append(word)

    print(f'Found {len(feature_word_d)} features shared by all vectors')
    selected_word = set()
    selected_feature_word_d = defaultdict(list)
    for f,ws in feature_word_d.items():
        if len(ws) >= int(bar):
            selected_feature_word_d[f]=ws
            for w in ws:
                selected_word.add(w)
    return selected_feature_word_d, selected_word

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", help="path of dataset")
    parser.add_argument("-threshold", help="minimal number of positive instances")
    parser.add_argument("-skipgram")
    parser.add_argument("-cbow")
    parser.add_argument("-word_topic_mask")
    parser.add_argument("-general_mask")
    parser.add_argument("-build_dir")

    args = parser.parse_args()
    working_dir = args.build_dir

    D_data = word_feature_dic(args.dataset)
    print(f'Loading {args.skipgram}')
    w2v1 = json.load(open(args.skipgram))
    print(f'Loading {args.cbow}')
    w2v2 = json.load(open(args.cbow))
    print(f'Loading {args.word_topic_mask}')
    wt_m = json.load(open(args.word_topic_mask))
    print(f'Loading {args.general_mask}')
    g_m = json.load(open(args.general_mask))

    D_fea_words,sel_w = selected_word_featrue(D_data,
                                        args.threshold,
                                        w2v1,
                                        w2v2,
                                        wt_m,
                                        g_m)
    print('the number of selected words:', len(sel_w))

    f1 = open(os.path.join(working_dir,'pos_train_data.txt'), 'w')
    f2 = open(os.path.join(working_dir,'pos_valid_data.txt'), 'w')
    f3 = open(os.path.join(working_dir,'pos_test_data.txt'), 'w')
    f4 = open(os.path.join(working_dir,'neg_train_data.txt'), 'w')
    f5 = open(os.path.join(working_dir,'neg_valid_data.txt'), 'w')
    f6 = open(os.path.join(working_dir,'neg_test_data.txt'), 'w')

    for key in D_fea_words.keys():
        pos = D_fea_words[key]
        neg = list(sel_w-set(pos))
        t_pos,test_pos = train_test_split(pos,random_state=2021,test_size=0.3)
        train_pos,valid_pos = train_test_split(t_pos,random_state=2021,test_size=0.25)
        t_neg,test_neg = train_test_split(neg,random_state=2021,test_size=0.3)
        train_neg,valid_neg = train_test_split(t_neg,random_state=2021,test_size=0.5)
        del t_pos
        del t_neg

        f1.write(key + '\t' + ', '.join(train_pos) + '\n')
        f2.write(key + '\t' + ', '.join(valid_pos) + '\n')
        f3.write(key + '\t' + ', '.join(test_pos) + '\n')
        f4.write(key + '\t' + ', '.join(train_neg) + '\n')
        f5.write(key + '\t' + ', '.join(valid_neg) + '\n')
        f6.write(key + '\t' + ', '.join(test_neg) + '\n')
        print(key, len(train_pos), len(valid_pos), len(test_pos), len(train_neg), len(valid_neg), len(test_neg))

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    f6.close()