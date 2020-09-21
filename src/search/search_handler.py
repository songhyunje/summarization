import logging
from datetime import datetime, timedelta

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q
from elasticsearch_dsl.query import MultiMatch, Match

logging.basicConfig(level=getattr(logging, 'INFO'))
logger = logging.getLogger(__name__)


class Searcher(object):
    def __init__(self, hosts, news_index, wiki_index=None):
        self.client = Elasticsearch(hosts=hosts)
        self.news_index = news_index
        self.wiki_index = wiki_index

    def search(self, query, category=None, from_date=None, to_date=None, count=False):
        from_datetime, to_datetime = self._covert_to_datetime(from_date, to_date)
        must = []
        if category and category != "ALL":
            must.append(Q('match', category=category))

        should = [Q('match', content=query), Q('match', title=query)]
        q = Q('bool', must=must, should=should, minimum_should_match=2)
        s = Search(using=self.client, index=self.news_index).query(q)
        if from_datetime and to_datetime:
            s = s.filter('range', publish_datetime={'from': from_datetime, 'to': to_datetime})

        s = s.sort({"publish_datetime": {'order': "desc"}})

        response = s.execute()
        if count:
            return response, s.count()

        return response

    def _covert_to_datetime(self, from_datetime, to_datetime):
        if not to_datetime:
            to_datetime = datetime.now()

        if not from_datetime:
            from_datetime = "2020-01-01"    # start_date!

        if not isinstance(from_datetime, datetime):
            from_datetime = datetime.strptime(from_datetime, '%Y-%m-%d')

        if not isinstance(to_datetime, datetime):
            to_datetime = datetime.strptime(to_datetime, '%Y-%m-%d')

        to_datetime = to_datetime.strftime("%Y-%m-%d %H:%M:%S")
        from_datetime = from_datetime.strftime("%Y-%m-%d %H:%M:%S")

        return from_datetime, to_datetime

    def count(self):
        news_res = self.client.count(index=self.news_index)
        logger.info("news_index: %s, news_count: %d" % (self.news_index, news_res['count']))
        return news_res['count']


