import argparse
import glob
import json
import logging
import os

import torch

from src.match_summary.match_selector import build_selector
from src.match_summary.matchsum import MatchSum

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint_dir",
    default=None,
    type=str,
    required=True,
    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--input", default='../../input.txt', type=str)
parser.add_argument("--result", default='result.txt', type=str)
parser.add_argument(
    "--min_length", default=50, type=int, help="Minimum number of tokens for the summaries.",
)
parser.add_argument(
    "--max_length", default=150, type=int, help="Maixmum number of tokens for the summaries.",
)
parser.add_argument(
    "--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.",
)

args = parser.parse_args()

logging.basicConfig(level=getattr(logging, 'INFO'))
logger = logging.getLogger(__name__)

checkpoints = list(sorted(glob.glob(os.path.join(args.checkpoint_dir, "checkpointepoch=*.ckpt"), recursive=True)))
logger.info("Load model from %s", checkpoints[-1])
logger.info("Device %s", args.device)
summarizer = MatchSum.load_from_checkpoint(checkpoints[-1], map_location=args.device)

tokenizer = summarizer.tokenizer
model = summarizer.model
model.to(args.device)
model.eval()

symbols = {
    "BOS": tokenizer.vocab["<!#s>"],
    "EOS": tokenizer.vocab["<!#/s>"],
    "PAD": tokenizer.vocab["<!#pad>"],
    "PERIOD": tokenizer.vocab["."],
}

selector = build_selector(args, tokenizer, symbols, model)

with open(args.input) as f:
    for text in f.readlines():  # each text is a line and a full story
        data = json.loads(text)
        src = data['src']
        target = data['tgt']

        sent_tokenized = tokenizer.encode_plus(src, max_length=512, pad_to_max_length=True,
                                               return_tensors="pt", device=args.device)

        cands_sentences, cands_input_ids, cands_attention_mask = [], [], []
        for cand in data['cands']:
            sent = ' '.join(cand)
            cand_tokenized = tokenizer.encode_plus(
                sent, max_length=512, pad_to_max_length=True, return_tensors="pt", device=args.device
            )
            cands_sentences.append(sent)
            cands_input_ids.append(cand_tokenized["input_ids"].squeeze())
            cands_attention_mask.append(cand_tokenized["attention_mask"].squeeze())

        sources = {"input_ids": sent_tokenized["input_ids"].squeeze(),
                   "attention_mask": sent_tokenized["attention_mask"].squeeze()}
        cands = {"input_ids": torch.stack(cands_input_ids),
                 "attention_mask": torch.stack(cands_attention_mask),
                 "sents": cands_sentences}

        batch = {'source': sent_tokenized, 'cands': cands}
        selected = selector.select(batch)
        logger.info("SOURCE: %s", src)
        logger.info("TARGET: %s", target)
        logger.info("PRED: %s", selected)
