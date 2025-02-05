import argparse
import glob
import logging
import os
import kss
from itertools import permutations

import torch

from src.match_summary.match_selector import build_selector
from src.match_summary.matchsum import MatchSum
from src.single_summary.singlesum import Summarization
from src.single_summary.translator import build_predictor

# DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DEVICE = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--single_checkpoint_dir",
    default=None,
    type=str,
    required=True)
parser.add_argument(
    "--match_checkpoint_dir",
    default=None,
    type=str,
    required=True)
parser.add_argument(
    "--max_src_seq_length",
    default=512,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer "
         "than this will be truncated, sequences shorter will be padded.",
)
parser.add_argument(
    "--max_tgt_seq_length",
    default=150,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer "
         "than this will be truncated, sequences shorter will be padded.",
)
parser.add_argument(
    "--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.",
)
parser.add_argument(
    "--alpha",
    default=0.95,
    type=float,
    help="The value of alpha for the length penalty in the beam search.",
)
parser.add_argument(
    "--beam_size", default=5, type=int, help="The number of beams to start with for each example.",
)
parser.add_argument("--input", default='data/multi/test.txt', type=str)
parser.add_argument(
    "--min_length", default=30, type=int, help="Minimum number of tokens for the summaries.",
)
parser.add_argument(
    "--max_length", default=100, type=int, help="Maixmum number of tokens for the summaries.",
)
parser.add_argument(
    "--block_trigram",
    action="store_true",
    help="Whether to block the existence of repeating trigrams in the text generated by beam search.",
)

args = parser.parse_args()
logging.basicConfig(level=getattr(logging, 'INFO'))
logger = logging.getLogger(__name__)

checkpoints = list(sorted(glob.glob(os.path.join(args.single_checkpoint_dir, "*.ckpt"), recursive=True)))
logger.info("Load model from %s", checkpoints[-1])
logger.info("Device %s", args.device)
summarizer = Summarization.load_from_checkpoint(checkpoints[-1], map_location=args.device)

tokenizer = summarizer.tokenizer
single_model = summarizer.model
single_model.to(args.device)
single_model.eval()

symbols = {
    "BOS": tokenizer.vocab["<!#s>"],
    "EOS": tokenizer.vocab["<!#/s>"],
    "PAD": tokenizer.vocab["<!#pad>"],
    "PERIOD": tokenizer.vocab["."],
}

translator = build_predictor(args, tokenizer, symbols, single_model)

checkpoints = list(sorted(glob.glob(os.path.join(args.match_checkpoint_dir, "checkpointepoch=*.ckpt"),
                                    recursive=True)))
logger.info("Load model from %s", checkpoints[-1])
logger.info("Device %s", args.device)
summarizer = MatchSum.load_from_checkpoint(checkpoints[-1], map_location=args.device)

match_model = summarizer.model
match_model.to(args.device)
match_model.eval()

selector = build_selector(args, tokenizer, symbols, match_model)


def candidates_from_single_summary(srcs):
    summary = []
    for src in srcs:
        sent_tokenized = tokenizer.encode_plus(src, max_length=512, pad_to_max_length=True,
                                               return_tensors="pt", device=args.device)
        for translate in translator.translate(sent_tokenized):
            pred = tokenizer.decode(translate)

        summary.extend(kss.split_sentences(pred))

    cands = list(permutations(summary, 2))
    cands += summary
    return cands


def select_summary(srcs, cands):
    srcs_sents, srcs_inpu_ids, srcs_attention_mask = [], [], []
    for doc in srcs:
        encoded_src = tokenizer.encode_plus(doc, max_length=512, pad_to_max_length=True,
                                            return_tensors="pt", device=args.device)
        srcs_sents.append(doc)
        srcs_inpu_ids.append(encoded_src["input_ids"].squeeze())
        srcs_attention_mask.append(encoded_src["attention_mask"].squeeze())

    cands_sents, cands_input_ids, cands_attention_mask = [], [], []
    for cand in cands:
        summary = ' '.join(cand)
        encoded_summary = tokenizer.encode_plus(
            summary, max_length=512, pad_to_max_length=True, return_tensors="pt", device=args.device
        )
        cands_sents.append(summary)
        cands_input_ids.append(encoded_summary["input_ids"].squeeze())
        cands_attention_mask.append(encoded_summary["attention_mask"].squeeze())

    srcs = {"input_ids": torch.stack(srcs_inpu_ids),
            "attention_mask": torch.stack(srcs_attention_mask)}
    cands = {"input_ids": torch.stack(cands_input_ids),
             "attention_mask": torch.stack(cands_attention_mask),
             "sents": cands_sents}

    batch = {'srcs': srcs, 'cands': cands}
    return selector.select_batches(batch)


with open(args.input) as f:
    for line in f:
        inputs = line.strip().split('|||||')
        srcs, target = inputs[:-1], inputs[-1]

        cands = candidates_from_single_summary(srcs)
        summary = select_summary(srcs, cands)
        for i, src in enumerate(srcs, 1):
            logger.info("SOURCE%d: %s", i, src)
        logger.info("TARGET: %s", target)
        logger.info("PRED: %s", summary)

