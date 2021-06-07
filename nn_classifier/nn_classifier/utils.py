import numpy as np
import os
import csv
from sklearn.metrics import precision_score, recall_score, f1_score
import re
from scipy.optimize import dual_annealing


def load_data(file):
    X = []
    Y = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            tmp = line.strip().split(',')
            X.append(tmp[0])
            Y.append(tmp[1:])
    return np.array(X), np.array(Y)



def load_prop_instances(file):
    feature_concept = {}
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            f, cns = line.strip().split('\t')
            feature_concept[f] = []
            for c in cns.split(', '):
                feature_concept[f].append(c.lower())
    return feature_concept


def load_nouns(file):
    nouns = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            nouns.append(line.strip().lower())
    return nouns


def load_pretrained_embeddings(file):
    embeddings_dict = {}
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0].lower()
            if word not in ['1840s', '1850s', '1860s', '1870s'] and not re.match(r'[A-Za-z]', line[0]):
                continue
            if word == 'nectarines':
                word = 'nectarine'
            vector = np.asarray(values[1:], "float")
            embeddings_dict[word] = vector
    return embeddings_dict


def load_layers_embedding(word_ls, embed_path)
    embeddings = defaultdict(lambda:defaultdict())
    for word in word_ls:
        try:
            vector_path = os.path.join(embed_path,word+'.txt')
            layer_count = 0
            with open(vector_path,'r') as inf:
                for line in inf:
                    vector = np.array(line.strip().split(' ')).astype(np.float)
                    embeddings[word][str(layer_count)] = vector.tolist()
                    layer_count+=1
        except:
            print(word, 'is missing from:', embed_path)
    print('total number of words in embedding:', len(embeddings.keys()))
    return embeddings


def pre_rec_f1(y_true, y_pred):
    return round(precision_score(y_true, y_pred), 4), round(recall_score(y_true, y_pred), 4), round(f1_score(y_true, y_pred), 4)


def word_embedding(embeddings, word_list):
    words_embeddings = []
    for ww in word_list:
        if ww in embeddings:
            words_embeddings.append(embeddings[ww])
        elif "missing_word" in embeddings:
            #print(ww)
            words_embeddings.append(embeddings['missing_word'])
        else:
            print(ww)
    return words_embeddings


def topic_mask(embedding, num_topic)
    mask_embedding = {}
    for w in embedding.keys():
        topics = embedding[w].keys()
        masks = []
        for i in range(num_topic):
            if str(i) in topics:
                masks.append(1)
            else:
                masks.append(0)
        mask_embedding[w] = np.array(masks)
    return mask_embedding


def write_csv(file, property, results):
    with open(file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['property', 'MAP', 'pre', 'rec', 'f1'])
        for i in range(len(property)):
            writer.writerow([property[i]] + results[i])
        writer.writerow(['mean'] + np.round(np.mean(np.array(results), axis=0), decimals=4).tolist())


def write_txt(file, results, header):
    output = header
    output += 'MAP\tpre\trec\tf1\n'
    output += '\t'.join([str(m) for m in np.round(np.mean(np.array(results), axis=0), decimals=4)]) + '\n'
    output += '\n\n'
    with open(file, 'a+', encoding='utf-8') as fout:
        fout.writelines(output)


def get_best_threshold(y_true, y_score, compare_symbol='greater'):
    ths = set(y_score)
    y_score = np.array(y_score)
    best_f1 = 0.0
    best_thd = 0.0
    for th in ths:
        y_pred = np.zeros(np.array(y_true).shape) - 1
        if compare_symbol == 'greater':
            y_pred[np.where(y_score >= th)[0]] = 1
        elif compare_symbol == 'less':
            y_pred[np.where(y_score <= th)[0]] = 1
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_thd = th
            best_f1 = f1
    #print(best_thd)
    return best_thd, best_f1


def load_bert_vectors(path, noun):
    data = []
    with open(os.path.join(path, noun +'.csv'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            vec = [float(l) for l in line]
            if vec not in data:
                data.append(vec)
    return data


def init_logging_path(log_path, task_name, file_name):
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


def f1(th, y_true, y_score):
    y_true = np.array(y_true)
    y_pred = np.zeros_like(y_true)
    y_pred[np.where(y_score > th)[0]] = 1
    f1 = f1_score(y_true, y_pred)
    return -f1


def optimal_threshold(y_true, y_score, Ns=20):
    bounds = [(np.min(y_score), np.max(y_score))]
    result = dual_annealing(f1, args=(y_true, y_score), bounds=bounds, maxiter=Ns)
    return result.x, -f1(result.x, y_true, y_score)