from math import log

from collections import defaultdict, Counter


def get_idf_dict(sents, tokenizer):
    idf_count = Counter()
    num_sents = len(sents)
    for sent in sents:
        # token_ids = tokenizer.encode(sent, add_special_tokens=False, max_length=max_length)
        # idf_count.update(set(token_ids))
        idf_count.update(set(tokenizer.tokenize(sent)))

    idf_dict = defaultdict(lambda: log((num_sents + 1) / 1))
    idf_dict.update({idx: log((num_sents + 1) / (c + 1)) for (idx, c) in idf_count.items()})
    return idf_dict


def memoize(func):
    cache = dict()

    def memoized_func(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return memoized_func
