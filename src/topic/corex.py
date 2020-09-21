import argparse
import io
import logging
import pickle
from os import listdir
from os.path import join

import scipy.sparse as ss
from corextopic import corextopic as ct
from sklearn.feature_extraction.text import CountVectorizer

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, 'INFO'))
logger = logging.getLogger(__name__)


def main(args):
    directory = args.directory
    logger.info("Loading from %s", directory)
    files = [join(directory, f) for f in listdir(directory)]
    corpus = []
    for fn in files:
        with io.open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                corpus.append(line.strip())
    logger.info("# of lines: %d", len(corpus))

    vectorizer = CountVectorizer(max_features=50000)
    vectorizer.fit(corpus)
    # X = vectorizer.fit_transform(corpus)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    X = vectorizer.transform(corpus)
    X = ss.csr_matrix(X)
    words = vectorizer.get_feature_names()
    logger.info(words)

    topic_model = ct.Corex(n_hidden=300)  # Define the number of latent (hidden) topics to use.
    topic_model.fit(X, words=words)

    # topic_model.fit(X, words=words,
    #                 anchors=['이강인', '손흥민', '도쿄올림픽', ['프로야구', 'kbo'], '코로나', '류현진', 'k리그', '김광현'],
    #                 anchor_strength=5)

    with open('anchor_topic.pkl', 'wb') as f:
        pickle.dump(topic_model, f)

    topics = topic_model.get_topics()
    for topic_n, topic in enumerate(topics):
        words, mis = zip(*topic)
        topic_str = str(topic_n + 1) + ': ' + ','.join(words)
        logger.info(topic_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Topic model')
    parser.add_argument('--directory', default='../../data/topic')
    args = parser.parse_args()
    main(args)
