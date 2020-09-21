import argparse
import io
import json
import logging
import re
import os
from datetime import datetime

import yaml
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, 'INFO'))
logger = logging.getLogger(__name__)

start_pattern = re.compile(r'^\[.*\]')
end_pattern = re.compile(r'(\(.*\)|\[.*\])$')
white = re.compile(r'\s+')


def load_news(fn):
    unique_news_ids = set()
    with io.open(fn, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for news in data['datas']:
            row_idx = news['rowIdx']
            news_id = news['aid']  # int(news['oid'] + news['aid'])

            unique_news_id = news['oid'] + news['aid']
            if unique_news_id in unique_news_ids:
                continue

            unique_news_ids.add(unique_news_id)

            title = news['newsTitle']
            title = re.sub(start_pattern, '', title).strip()
            title = re.sub(end_pattern, '', title).strip()

            final_date = news['finalDate']
            final_time = news['finalTime'].replace("오후", "PM").replace("오전", "AM")

            date_string = "%s %s" % (final_date, final_time)
            publish_datetime = datetime.strptime(date_string, '%Y-%m-%d %p %I:%M').strftime("%Y-%m-%d %H:%M:%S")

            naver_url = news['naverUrl']
            origin_url = news['originUrl']
            publisher = news['officeName']
            category = news['category']
            if category == '해외야구':
                category = "IBASEBALL"
            elif category == '해외축구':
                category = "ISOCCER"
            elif category == '야구':
                category = "BASEBALL"
            elif category == '연예':
                category = "ENTERTAINMENT"
            elif category == '축구':
                category = "SOCCER"
            elif category == '배구':
                category = "VOLLEYBALL"
            elif category == '농구':
                category = "BASKETBALL"
            elif category == '골프':
                category = "GOLF"

            content = news['extContent']  # replace_all(news['extContent'], replace_dict)
            content = re.sub(white, ' ', content).strip()
            yield {
                '_id': row_idx, 'news_id': news_id, 'type': 'news',
                'title': title, 'naver_url': naver_url, 'origin_url': origin_url,
                'category': category, 'publisher': publisher, 'publish_datetime': publish_datetime,
                'content': content
            }


def main(args):
    with open(args.config, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.BaseLoader)

    client = Elasticsearch(hosts=cfg['elasticsearch']['host'])
    index = cfg['elasticsearch']['news_index']

    files = [f for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f))]
    for fin in files:
        logger.info(fin)
        for ok, result in streaming_bulk(client, load_news(os.path.join(args.input, fin)), index=index, chunk_size=100):
            action, result = result.popitem()
            doc_id = "/%s/doc/%s" % (index, result["_id"])

            if not ok:
                logger.warning("Failed to %s document %s: %r" % (action, doc_id, result))
            else:
                logger.info(doc_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Indexing news.')
    parser.add_argument('--config', default='config.yaml', help='Config file.')
    parser.add_argument('--input', default='data/search/2020', help='News json file.')
    args = parser.parse_args()
    main(args)
