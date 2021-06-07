import logging
import gensim
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim import corpora, models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", help="number of topics")
    parser.add_argument("-alpha", help="alpha Hyperparameters",default=0.0001)
    parser.add_argument("-wordids", help="location of wordids.txt")
    parser.add_argument("-tfidf", help="location of tfidf.mm")
    parser.add_argument("-workers", help="number of workers, if multi-processing is required")
    parser.add_argument("-bow", help="location of bow.mm.metadata.cpickle")
    parser.add_argument("-build_dir", help="location of output dir")
    args = parser.parse_args()


    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # load id->word mapping (the dictionary)
    id2word = gensim.corpora.Dictionary.load_from_text(args.wordids)
    # load corpus iterator
    mm = gensim.corpora.MmCorpus(args.tfidf)

    lda_model = gensim.models.LdaMulticore(corpus=mm,
                                            id2word=id2word,
                                            num_topics=args.k,
                                            alpha=np.ones(args.k)*args.alpha,
                                            random_state=2021,
                                            workers=args.workers,
                                               #update_every=1,
                                            passes=1)

    """mapping between title and its id"""

    title_num_mapping = defaultdict(str)
    with open(args.bow, 'rb') as meta_file:
        docno2metadata = pickle.load(meta_file)
    for num in range(len(docno2metadata)):
        title = docno2metadata[num][1]
        title_num_mapping[title]=num
    
    """Mapping between title and its topic distributions"""

    count = 0
    title_distribution_mapping = defaultdict(list)
    for title,doc_num in title_num_mapping.items():
        vec = mm[doc_num]
        distribution = lda_model.get_document_topics(vec)
        title_distribution_mapping[title]=distribution
        count = len(title_distribution_mapping.keys())
        if count%100000==0:
            print('completed ',count,'distributions')
            
    with open(os.path.join(args.build_dir,'title_distribution_mapping.pkl'), 'wb') as f:
        pickle.dump(title_distribution_mapping, f)
