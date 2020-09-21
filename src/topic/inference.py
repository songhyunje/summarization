import argparse
import logging
import pickle

import numpy as np
import scipy.sparse as ss
from pynori.korean_analyzer import KoreanAnalyzer

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, 'INFO'))
logger = logging.getLogger(__name__)

tags = ["E", "IC", "J", "MAG", "MM", "SC", "SE", "SF", "SN", "SP", "SSC",
        "SSO", "SY", "UNA", "UNKNOWN", "VA", "VCN", "VCP", "VSV", "VV",
        "VX", "XPN", "XR", "XSA", "XSN", "XSV"]


def main(args):
    nori = KoreanAnalyzer(decompound_mode='MIXED',  # DISCARD or MIXED or NONE
                          infl_decompound_mode='NONE',  # DISCARD or MIXED or NONE
                          discard_punctuation=True,
                          output_unknown_unigrams=False,
                          pos_filter=True, stop_tags=tags,
                          synonym_filter=False, mode_synonym='NORM')  # NORM or EXTENSION

    logger.info("Loading vectorizer from %s", args.vectorizer)
    vectorizer = pickle.load(open(args.vectorizer, 'rb'))

    logger.info("Loading topic model from %s", args.model)
    topic_model = pickle.load(open(args.model, 'rb'))

    topics = topic_model.get_topics()
    for topic_n, topic in enumerate(topics):
        words, mis = zip(*topic)
        topic_str = str(topic_n + 1) + ': ' + ','.join(words)
        logger.info(topic_str)

    sents = [
        "손흥민(28·토트넘·사진)이 영국으로 떠난 지 한 달도 안돼 다시 귀국했다."
    ]

    tokenized_sents = []
    for sent in sents:
        tokens = []
        results = nori.do_analysis(sent)
        for offset, term, pos in zip(results['offsetAtt'], results['termAtt'], results['posTagAtt']):
            if pos.startswith('E') or pos.startswith('J'):
                continue
            tokens.append(sent[offset[0]:offset[1]])

        tokenized_sents.append(' '.join(tokens))
        logger.info('Sentence: ' + sent)
        logger.info('Tokenized: ' + ' '.join(tokens))

    X = vectorizer.transform(tokenized_sents)
    X = ss.csr_matrix(X)

    preds = topic_model.predict(X)
    for pred in preds:
        print(pred)
        result = np.where(pred)
        for idx in result[0].tolist():
            words, mis = zip(*topics[idx])
            topic_str = str(idx + 1) + ': ' + ','.join(words)
            logger.info(topic_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Topic model')
    parser.add_argument('--vectorizer', default='vectorizer.pkl')
    parser.add_argument('--model', default='anchor_topic.pkl')
    args = parser.parse_args()
    main(args)
