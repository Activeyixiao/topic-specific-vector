import sys
import os
import json
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))
import argparse
from src.utils import *
from src.train_test import *
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", help=" path of dataset", choices=['MC', 'CSLB', 'SS','BD'])
    parser.add_argument("-vector_type", help="vector type", choices=['topic', 'layers', 'single']) # 'layers': weighted-average of word vector from 12/24 layer of language model; 'single': word vectors in which word only have single vector
    parser.add_argument("-embed_path", help="path of word vector embedding")
    parser.add_argument("-mask", help="whether to use masked vector", action="store_true", default=True)
    parser.add_argument("-batch_size", help="batch size", default=16)
    parser.add_argument("-in_features", help="dimension of input features",choices=[768,1024,300],type=int)
    parser.add_argument("-learning_rate", help="learning rate", default=0.001, type=float)
    parser.add_argument("-weight_decay", help="weight decay", default=0.02, type=float)
    parser.add_argument("-seed", help="seed", default=2021)
    parser.add_argument("-gpu", help="whether to use gpu", default=False, action="store_true")
    parser.add_argument("-num_epoch", help="number of epoches to train the model", default=100, type=int)
    parser.add_argument("-vector_name", help="type of the word vector", choices=['bert-base', 'bert-large', 'roberta-base','roberta-large', 'SKIPGRAM','CBOW'])
    args = parser.parse_args()

    result_path = os.path.join(os.path.abspath(project_path), 'result_net', args.dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    log_file_path = init_logging_path(os.path.join(os.getcwd(), 'log'), 'net', "net")
    logging.basicConfig(filename=log_file_path,
                        level=10,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info(args)

    logging.info("loading property and corresponding instances")

    data_path = os.path.join(project_path, "data", args.dataset)

    pos_train = load_prop_instances(os.path.join(data_path, f'pos_train_data.txt'))
    neg_train = load_prop_instances(os.path.join(data_path, f"neg_train_data.txt"))
    pos_test = load_prop_instances(os.path.join(data_path, f"pos_test_data.txt"))
    neg_test = load_prop_instances(os.path.join(data_path, f"neg_test_data.txt"))
    pos_valid = load_prop_instances(os.path.join(data_path, f"pos_valid_data.txt"))
    neg_valid = load_prop_instances(os.path.join(data_path, f"neg_valid_data.txt"))
    word_ls = [line.strip() for line in open(os.path.join(data_path,'all_words.txt'),'r')]
    use_topic = False
    if args.vector_type == 'topic':
        embeddings = json.load(open(args.embed_path))
        embeddings,mask_embedding = word_topic_embedding(embeddings,word_ls,25,int(args.in_features))
        use_topic = True
        num_layers = 25
    elif args.vector_type == 'layers':
        embeddings = load_layers_embedding(word_ls,args.embed_path)
        if args.in_features == 1024:
            num_layers = 24
        elif args.in_features == 768:
            num_layers = 12
        mask_embedding = {}
        for w in word_ls:
            mask_embedding[w]=num_layers*[1]
        use_topic = True
    else:
        embeddings = json.load(open(args.embed_path))


    logging.info('start to train and test')
    properties = []
    results = []
    cnt = 1
    for prop in pos_train:
        logging.info(str(cnt) + ', ' + prop)

        pos_train_data = word_embedding(embeddings, pos_train[prop],topic=use_topic)
        neg_train_data = word_embedding(embeddings, neg_train[prop],topic=use_topic)
        pos_test_data = word_embedding(embeddings, pos_test[prop],topic=use_topic)
        neg_test_data = word_embedding(embeddings, neg_test[prop],topic=use_topic)
        pos_valid_data = word_embedding(embeddings, pos_valid[prop],topic=use_topic)
        neg_valid_data = word_embedding(embeddings, neg_valid[prop],topic=use_topic)
        if use_topic:
            pos_train_mask = word_embedding(mask_embedding,pos_train[prop])
            neg_train_mask = word_embedding(mask_embedding,neg_train[prop])
            pos_test_mask = word_embedding(mask_embedding,pos_test[prop])
            neg_test_mask = word_embedding(mask_embedding,neg_test[prop])
            pos_valid_mask = word_embedding(mask_embedding, pos_valid[prop])
            neg_valid_mask = word_embedding(mask_embedding, neg_valid[prop])
            train_mask = np.array(pos_train_mask + neg_train_mask)
            test_mask = np.array(pos_test_mask + neg_test_mask)
            valid_mask = np.array(pos_valid_mask + neg_valid_mask)

        train_data = np.array(pos_train_data + neg_train_data)
        train_label = np.array([1] * len(pos_train_data) + [0] * len(neg_train_data))
        test_data = np.array(pos_test_data + neg_test_data)
        test_label = np.array([1] * len(pos_test_data) + [0] * len(neg_test_data))
        valid_data = np.array(pos_valid_data + neg_valid_data)
        valid_label = np.array([1] * len(pos_valid_data) + [0] * len(neg_valid_data))

        del pos_train_data
        del neg_train_data
        del pos_test_data
        del neg_test_data
        del pos_valid_data
        del neg_valid_data

        model_path = os.path.join(project_path, "trained_model", args.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_fname = '{0:s}/{1:s}.pt'.format(model_path, prop + "_" + args.vector_type)

        logging.info("train model")
        if use_topic:
            train_topic(args, num_layers, model_fname, train_data, valid_data, train_label, valid_label, train_mask, valid_mask)
        else:
            train(args, model_fname, train_data, valid_data, train_label, valid_label)

        logging.info("reload trained model")
        if use_topic:
            model = Binary_topic_Net(args, num_layers)
        else:
            model = Classifier(args)
        model.load_state_dict(torch.load(model_fname, map_location=torch.device('cpu')))

        logging.info("choose threshold")
        if use_topic:
            th = test_topic(args, valid_data, valid_mask, valid_label, model, eval_mark="dev")
        else:
            th = test(args, valid_data, valid_label, model, eval_mark="dev")

        logging.info("test data")
        if use_topic:
            rr = test_topic(args, test_data, test_mask, test_label, model, th, eval_mark="test")
        else:
            rr = test(args, test_data, test_label, model, th, eval_mark="test")

        results.append(rr)
        properties.append(prop)
        cnt += 1


    csv_file = os.path.join(result_path, args.vector_type +'_'+ args.vector_name + '_'+str(args.batch_size)+'_'+str(args.learning_rate)+'_.csv')
    header = 'svc, ' + args.vector_type + ',' + args.vector_name+'\n'

    write_csv(csv_file, properties, results)
    write_txt(os.path.join(os.path.abspath(project_path), 'result_net', args.dataset + '.txt'), results, header)
