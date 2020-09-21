import logging
from argparse import Namespace
from itertools import permutations

import kss
import torch
import yaml

from src.common.score.bert_scorer import BertScorer
from src.common.score.rouge_scorer import RougeScorer
from src.match_summary.match_selector import build_selector as match_selector
from src.match_summary.matchsum import MatchSum
from src.multi_summary.multisum import MDS
from src.multi_summary.translator import build_predictor as multi_predictor
from src.single_summary.singlesum import Summarization
from src.single_summary.translator import build_predictor as single_predictor

logging.basicConfig(level=getattr(logging, 'INFO'))
logger = logging.getLogger(__name__)

with open('config.yaml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.BaseLoader)

singlesum_model = cfg['singlesum']['model']
singlesum_device = cfg['singlesum']['device']
single_summarizer = Summarization.load_from_checkpoint(singlesum_model, map_location=singlesum_device)

single_model = single_summarizer.model.to(singlesum_device)
single_model.eval()

tokenizer = single_summarizer.tokenizer
symbols = {
    "BOS": tokenizer.vocab["<!#s>"],
    "EOS": tokenizer.vocab["<!#/s>"],
    "PAD": tokenizer.vocab["<!#pad>"],
    "PERIOD": tokenizer.vocab["."],
}

multisum_model = cfg['multisum']['model']
multisum_device = cfg['multisum']['device']
multi_summarizer = MDS.load_from_checkpoint(multisum_model, map_location=multisum_device)

multi_model = multi_summarizer.model.to(multisum_device)
multi_model.eval()

matchsum_model = cfg['matchsum']['model']
matchsum_device = cfg['matchsum']['device']
match_summarizer = MatchSum.load_from_checkpoint(matchsum_model, map_location=matchsum_device)

match_model = match_summarizer.model.to(matchsum_device)
match_model.eval()

single_param = {'alpha': float(cfg['singlesum']['alpha']),
                'block_trigram': cfg['singlesum']['block_trigram'],
                'beam_size': int(cfg['singlesum']['beam_size']),
                'min_length': int(cfg['singlesum']['min_length']),
                'max_length': int(cfg['singlesum']['max_length']),
                }
multi_param = {'alpha': float(cfg['multisum']['alpha']),
               'block_trigram': cfg['multisum']['block_trigram'],
               'beam_size': int(cfg['multisum']['beam_size']),
               'min_length': int(cfg['multisum']['min_length']),
               'max_length': int(cfg['multisum']['max_length']),
               }
match_param = {'min_length': int(cfg['multisum']['min_length']),
               'max_length': int(cfg['multisum']['max_length']),
               }

single_translator = single_predictor(Namespace(**single_param), tokenizer, symbols, single_model)
multi_translator = multi_predictor(Namespace(**multi_param), tokenizer, symbols, multi_model)
match_selector = match_selector(Namespace(**match_param), tokenizer, symbols, match_model)


def convert_to_features(docs, device='cuda'):
    features = []
    for doc in docs:
        doc_repr = tokenizer.encode_plus(doc, max_length=512, pad_to_max_length=True, return_tensors="pt")
        doc_repr['input_ids'] = doc_repr['input_ids'].to(device)
        doc_repr['attention_mask'] = doc_repr['attention_mask'].to(device)
        doc_repr['sentence'] = doc
        features.append(doc_repr)
    return features


def singlesum(features, min_len=None, max_len=None):
    # logger.info("min_len: %d, max_len: %d" % (min_len, max_len))
    for translate in single_translator.translate(features, min_len, max_len):
        pred = tokenizer.decode(translate)

    return pred


def multisum(features: list, min_len=None, max_len=None):
    # logger.info("min_len: %d, max_len: %d", min_len, max_len)
    for translate in multi_translator.translate(features, min_len, max_len):
        pred = tokenizer.decode(translate)

    return pred


def matchsum(srcs, cands):
    srcs_sents, srcs_inpu_ids, srcs_attention_mask = [], [], []
    for doc in srcs:
        encoded_src = tokenizer.encode_plus(doc, max_length=512, pad_to_max_length=True, return_tensors="pt")
        srcs_sents.append(doc)
        srcs_inpu_ids.append(encoded_src["input_ids"].squeeze())
        srcs_attention_mask.append(encoded_src["attention_mask"].squeeze())

    cands_sents, cands_input_ids, cands_attention_mask = [], [], []
    for cand in cands:
        summary = ' '.join(cand)
        encoded_summary = tokenizer.encode_plus(summary, max_length=512, pad_to_max_length=True, return_tensors="pt")
        cands_sents.append(summary)
        cands_input_ids.append(encoded_summary["input_ids"].squeeze())
        cands_attention_mask.append(encoded_summary["attention_mask"].squeeze())

    srcs = {"input_ids": torch.stack(srcs_inpu_ids),
            "attention_mask": torch.stack(srcs_attention_mask)}
    cands = {"input_ids": torch.stack(cands_input_ids),
             "attention_mask": torch.stack(cands_attention_mask),
             "sents": cands_sents}

    batch = {'srcs': srcs, 'cands': cands}
    return match_selector.select_batches(batch)


if __name__ == '__main__':
    rouge_scorer = RougeScorer(['rouge1', 'rougeL'])
    bert_scorer = BertScorer()

    single_sum_result, single_sum_avg_result, multi_sum_result, match_sum_result = [], [], [], []
    with open('data/multi/test.txt') as f:
        for i, line in enumerate(f, 1):
            inputs = line.strip().split('|||||')
            docs, target = inputs[:-1], inputs[-1]
            features = convert_to_features(docs)

            multi_summary = multisum(features)

            single_summaries, single_sents = [], []
            for feature in features:
                pred = singlesum(feature)
                single_summaries.append(pred)
                single_sents.extend(kss.split_sentences(pred))

            cands = list(permutations(single_sents, 2))
            cands += single_sents
            match_summary = matchsum(docs, cands)

            single_sum_rouge1_scores, single_sum_rougeL_scores, single_sum_bert_scores = [], [], []

            for single_sum in single_summaries:
                rouge_score = rouge_scorer.score(single_sum, target)
                single_sum_rouge1_scores.append(rouge_score['rouge1'].fmeasure)
                single_sum_rougeL_scores.append(rouge_score['rougeL'].fmeasure)
                single_sum_bert_scores.append(bert_scorer.score([single_sum], [target])[0])

            single_sum_result.append((max(single_sum_rouge1_scores),
                                      max(single_sum_rougeL_scores),
                                      max(single_sum_bert_scores)))
            single_sum_avg_result.append((sum(single_sum_rouge1_scores) / len(single_sum_rouge1_scores),
                                          sum(single_sum_rougeL_scores) / len(single_sum_rougeL_scores),
                                          sum(single_sum_bert_scores) / len(single_sum_bert_scores)))
            multi_sum_rouge_score = rouge_scorer.score(multi_summary, target)
            multi_sum_result.append((multi_sum_rouge_score['rouge1'].fmeasure, multi_sum_rouge_score['rougeL'].fmeasure,
                                     bert_scorer.score([multi_summary], [target])[0]))
            match_sum_rouge_score = rouge_scorer.score(match_summary, target)
            match_sum_result.append((match_sum_rouge_score['rouge1'].fmeasure, match_sum_rouge_score['rougeL'].fmeasure,
                                     bert_scorer.score([match_summary], [target])[0]))

    single_sum_mean_rouge1 = sum([pair[0] for pair in single_sum_result]) / len(single_sum_result)
    single_sum_mean_rougeL = sum([pair[1] for pair in single_sum_result]) / len(single_sum_result)
    single_sum_mean_bert = sum([pair[2] for pair in single_sum_result]) / len(single_sum_result)

    print(single_sum_mean_rouge1, single_sum_mean_rougeL, single_sum_mean_bert)

    single_sum_avg_mean_rouge1 = sum([pair[0] for pair in single_sum_avg_result]) / len(single_sum_avg_result)
    single_sum_avg_mean_rougeL = sum([pair[1] for pair in single_sum_avg_result]) / len(single_sum_avg_result)
    single_sum_avg_mean_bert = sum([pair[2] for pair in single_sum_avg_result]) / len(single_sum_avg_result)

    print(single_sum_avg_mean_rouge1, single_sum_avg_mean_rougeL, single_sum_avg_mean_bert)

    multi_sum_mean_rouge1 = sum([pair[0] for pair in multi_sum_result]) / len(multi_sum_result)
    multi_sum_mean_rougeL = sum([pair[1] for pair in multi_sum_result]) / len(multi_sum_result)
    multi_sum_mean_bert = sum([pair[2] for pair in multi_sum_result]) / len(multi_sum_result)

    print(multi_sum_mean_rouge1, multi_sum_mean_rougeL, multi_sum_mean_bert)

    match_sum_mean_rouge1 = sum([pair[0] for pair in match_sum_result]) / len(match_sum_result)
    match_sum_mean_rougeL = sum([pair[1] for pair in match_sum_result]) / len(match_sum_result)
    match_sum_mean_bert = sum([pair[2] for pair in match_sum_result]) / len(match_sum_result)

    print(match_sum_mean_rouge1, match_sum_mean_rougeL, match_sum_mean_bert)
