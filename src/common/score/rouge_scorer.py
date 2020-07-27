import re

import six
from nltk.tokenize import WhitespaceTokenizer
from rouge_score import scoring
from rouge_score.rouge_scorer import _score_lcs, _summary_level_lcs, _create_ngrams, _score_ngrams


class RougeScorer(scoring.BaseScorer):
    def __init__(self, rouge_types, tokenizer=WhitespaceTokenizer()):
        self.rouge_types = rouge_types
        self.tokenizer = tokenizer

    def score(self, target, prediction):
        target_tokens = self.tokenizer.tokenize(target)
        prediction_tokens = self.tokenizer.tokenize(prediction)
        result = {}

        for rouge_type in self.rouge_types:
            if rouge_type == "rougeL":
                # Rouge from longest common subsequences.
                scores = _score_lcs(target_tokens, prediction_tokens)
            elif rouge_type == "rougeLsum":
                # Note: Does not support multi-line text.
                def get_sents(text):
                    # Assume sentences are separated by newline.
                    sents = six.ensure_str(text).split("\n")
                    sents = [x for x in sents if len(x)]
                    return sents

                target_tokens_list = [self.tokenizer.tokenize(s) for s in get_sents(target)]
                prediction_tokens_list = [self.tokenizer.tokenize(s) for s in get_sents(prediction)]
                scores = _summary_level_lcs(target_tokens_list, prediction_tokens_list)
            elif re.match(r"rouge[0-9]$", six.ensure_str(rouge_type)):
                # Rouge from n-grams.
                n = int(rouge_type[5:])
                if n <= 0:
                    raise ValueError("rougen requires positive n: %s" % rouge_type)
                target_ngrams = _create_ngrams(target_tokens, n)
                prediction_ngrams = _create_ngrams(prediction_tokens, n)
                scores = _score_ngrams(target_ngrams, prediction_ngrams)
            else:
                raise ValueError("Invalid rouge type: %s" % rouge_type)
            result[rouge_type] = scores

        return result
