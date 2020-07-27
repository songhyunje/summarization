import json
import torch
import logging
import os
import numpy as np

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SummarizationDataset(Dataset):
    def __init__(self, tokenizer, data_dir="./data", type_path="train",
                 max_src_seq_length=512, max_tgt_seq_length=120,
                 use_token_type=False, sample=None):
        super(SummarizationDataset).__init__()
        self.tokenizer = tokenizer
        self.use_token_type = use_token_type

        self.source = []
        self.target = []

        logging.info("Loading " + type_path + " txt")
        with open(os.path.join(data_dir, type_path + ".txt"), "r") as f:
            for text in f.readlines():  # each text is a line and a full story
                src, tgt = text.split("|||||")
                src_tokenized = tokenizer.encode_plus(
                    src, max_length=max_src_seq_length, pad_to_max_length=True, return_tensors="pt"
                )
                tgt_tokenized = tokenizer.encode_plus(
                    tgt, max_length=max_tgt_seq_length, pad_to_max_length=True, return_tensors="pt"
                )
                self.source.append(src_tokenized)
                self.target.append(tgt_tokenized)

                if sample and len(self.source) > sample:
                    break

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        src_mask = self.source[index]["attention_mask"].squeeze()
        tgt_mask = self.target[index]["attention_mask"].squeeze()

        if self.use_token_type:
            src_type = self.source[index]["token_type_ids"].squeeze()
            return {"source_ids": source_ids, "source_type": src_type, "source_mask": src_mask,
                    "target_ids": target_ids, "target_mask": tgt_mask}
        else:
            return {"source_ids": source_ids, "source_mask": src_mask,
                    "target_ids": target_ids, "target_mask": tgt_mask}


class MDSDataset(Dataset):
    def __init__(self, tokenizer, data_dir="./data/multi", type_path="train",
                 max_src_seq_length=1024, max_src_sent_length=300,
                 max_tgt_seq_length=120, sample=None):
        super(MDSDataset).__init__()
        self.source = []
        self.target = []

        logging.info("Loading " + type_path + " txt")
        with open(os.path.join(data_dir, type_path + ".txt"), "r") as f:
            for text in f.readlines():  # each text is a line and a full story
                tokens = text.split("|||||")
                src_tokenized = tokenizer.encode_plus_mds(
                    tokens[:-1], max_sent_length=max_src_sent_length, max_length=max_src_seq_length,
                    pad_to_max_length=True, return_tensors="pt"
                )
                tgt_tokenized = tokenizer.encode_plus(
                    tokens[-1], max_length=max_tgt_seq_length, pad_to_max_length=True, return_tensors="pt"
                )
                self.source.append(src_tokenized)
                self.target.append(tgt_tokenized)
                if sample and len(self.source) > sample:
                    break

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        src_mask = self.source[index]["attention_mask"].squeeze()
        tgt_mask = self.target[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": tgt_mask}


class MatchSumDataset(Dataset):
    def __init__(self, tokenizer, data_dir="./data/match", type_path="train",
                 max_src_seq_length=512, max_tgt_seq_length=300, sample=None):
        super(MatchSumDataset).__init__()
        self.tokenizer = tokenizer

        self.source = []
        self.target = []
        self.cands = []
        self.scores = []

        logging.info("Loading " + type_path + " json")
        with open(os.path.join(data_dir, type_path + ".json"), "r") as f:
            for text in f.readlines():  # each text is a line and a full story
                data = json.loads(text)
                src = data['src']
                tgt = data['tgt']

                src_tokenized = tokenizer.encode_plus(
                    src, max_length=max_src_seq_length, pad_to_max_length=True, return_tensors="pt"
                )
                tgt_tokenized = tokenizer.encode_plus(
                    tgt, max_length=max_tgt_seq_length, pad_to_max_length=True, return_tensors="pt"
                )
                self.source.append(src_tokenized)
                self.target.append(tgt_tokenized)

                cands = []
                for cand in data['cands']:
                    cand_tokenized = tokenizer.encode_plus(
                        '. '.join(cand), max_length=max_tgt_seq_length, pad_to_max_length=True, return_tensors="pt"
                    )
                    cands.append(cand_tokenized)
                self.cands.append(cands)
                self.scores.append([np.mean(score) for score in data['scores']])

                if sample and len(self.source) > sample:
                    break

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        sources = {"input_ids": self.source[index]["input_ids"].squeeze(),
                   "attention_mask": self.source[index]["attention_mask"].squeeze()}
        target = {"input_ids": self.target[index]["input_ids"].squeeze(),
                  "attention_mask": self.target[index]["attention_mask"].squeeze()}

        cands_input_ids = []
        cands_attention_mask = []
        for cand in self.cands[index]:
            cands_input_ids.append(cand["input_ids"].squeeze())
            cands_attention_mask.append(cand["attention_mask"].squeeze())

        cands = {"input_ids": torch.stack(cands_input_ids),
                 "attention_mask": torch.stack(cands_attention_mask),
                 "scores": self.scores[index]}

        return {"source": sources, "target": target, "cands": cands}


class MatchSumPredictDataset(Dataset):
    def __init__(self, tokenizer, data_dir="./data/match", type_path="test",
                 max_src_seq_length=512, max_tgt_seq_length=300, sample=None):
        super(MatchSumDataset).__init__()
        self.tokenizer = tokenizer

        self.source = []
        self.cands = []

        logging.info("Loading " + type_path + " json")
        with open(os.path.join(data_dir, type_path + ".json"), "r") as f:
            for text in f.readlines():  # each text is a line and a full story
                data = json.loads(text)
                src = data['src']

                src_tokenized = tokenizer.encode_plus(
                    src, max_length=max_src_seq_length, pad_to_max_length=True, return_tensors="pt"
                )
                src_tokenized["sentence"] = src
                self.source.append(src_tokenized)

                cands = []
                for cand in data['cands']:
                    cand_tokenized = tokenizer.encode_plus(
                        '. '.join(cand), max_length=max_tgt_seq_length, pad_to_max_length=True, return_tensors="pt"
                    )
                    cand_tokenized['sentence'] = '. '.join(cand)
                    cands.append(cand_tokenized)
                self.cands.append(cands)

                if sample and len(self.source) > sample:
                    break

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        sources = {"input_ids": self.source[index]["input_ids"].squeeze(),
                   "attention_mask": self.source[index]["attention_mask"].squeeze()}

        cands_sentences, cands_input_ids, cands_attention_mask = [], [], []
        for cand in self.cands[index]:
            cands_sentences.append(cand["sentence"])
            cands_input_ids.append(cand["input_ids"].squeeze())
            cands_attention_mask.append(cand["attention_mask"].squeeze())

        cands = {"input_ids": torch.stack(cands_input_ids),
                 "attention_mask": torch.stack(cands_attention_mask)}

        return {"src_sent": self.source[index]["sentence"],
                "cand_sents": cands_sentences,
                "source": sources, "cands": cands}
