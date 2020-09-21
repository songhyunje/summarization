import logging
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from src.common.dataset.dataset import MatchSumDataset
from src.common.model.modeling import BertModel
from src.common.tokenization.tokenization_kt import Tokenizer

logger = logging.getLogger(__name__)


class MatchSum(pl.LightningModule):
    def __init__(self, hparams):
        super(MatchSum, self).__init__()
        self.hparams = hparams
        self.tokenizer = Tokenizer(os.path.join(self.hparams.model_name_or_path,
                                                self.hparams.vocab_name), max_len=512)

        model_state_dict = torch.load(os.path.join(self.hparams.model_name_or_path,
                                                   self.hparams.model_name))
        self.model = BertModel.from_pretrained(self.hparams.model_name_or_path, **model_state_dict)

    def prepare_data(self):
        # load_and cache_data
        train_dataset = self.load_and_cache_data(data_type="train")
        val_dataset = self.load_and_cache_data(data_type="val")

        # assign to use in dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def forward(self, batch):
        src_embeddings = self.model(input_ids=batch["source"]["input_ids"],
                                    attention_mask=batch["source"]["attention_mask"],
                                    output_all_encoded_layers=False)
        tgt_embeddings = self.model(input_ids=batch["target"]["input_ids"],
                                    attention_mask=batch["target"]["attention_mask"],
                                    output_all_encoded_layers=False)
        batch_size, candidate_num, input_len = batch["cands"]["input_ids"].size()
        cands_embeddings = self.model(input_ids=batch["cands"]["input_ids"].view(-1, input_len),
                                      attention_mask=batch["cands"]["attention_mask"].view(-1, input_len),
                                      output_all_encoded_layers=False)

        doc_emb = src_embeddings[:, 0, :]
        sum_emb = tgt_embeddings[:, 0, :]
        summary_score = torch.cosine_similarity(sum_emb, doc_emb, dim=-1)

        candidate_emb = cands_embeddings[:, 0, :].view(batch_size, candidate_num, -1)
        doc_emb = doc_emb.unsqueeze(1).expand_as(candidate_emb)
        score = torch.cosine_similarity(candidate_emb, doc_emb, dim=-1)  # batch x candidate_num
        return score, summary_score

    def predict(self, batch):
        src_embeddings = self.model(input_ids=batch["source"]["input_ids"],
                                    attention_mask=batch["source"]["attention_mask"],
                                    output_all_encoded_layers=False)
        batch_size, candidate_num, input_len = batch["cands"]["input_ids"].size()
        cands_embeddings = self.model(input_ids=batch["cands"]["input_ids"].view(-1, input_len),
                                      attention_mask=batch["cands"]["attention_mask"].view(-1, input_len),
                                      output_all_encoded_layers=False)

        doc_emb = src_embeddings[:, 0, :]
        candidate_emb = cands_embeddings[:, 0, :].view(batch_size, candidate_num, -1)
        doc_emb = doc_emb.unsqueeze(1).expand_as(candidate_emb)
        score = torch.cosine_similarity(candidate_emb, doc_emb, dim=-1)
        return score

    def is_logger(self):
        return self.trainer.proc_rank <= 0

    def match_loss(self, score, summary_score, margin=0.01):
        loss = 0
        # candidate loss  (candidate sorted by score)
        n = score.size(1)
        for i in range(1, n - 1):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones(pos_score.size()).cuda(score.device)
            loss += F.margin_ranking_loss(pos_score, neg_score, ones, margin * i)

        # summary loss
        pos_score = summary_score.unsqueeze(-1).expand_as(score)
        neg_score = score
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        ones = torch.ones(pos_score.size()).to(score.device)
        loss += F.margin_ranking_loss(pos_score, neg_score, ones, margin=0.0)
        return loss

    def training_step(self, batch, batch_idx):
        score, summary_score = self(batch)
        loss = self.match_loss(score, summary_score)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        score, summary_score = self(batch)
        loss = self.match_loss(score, summary_score)

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        tqdm_dict = {}
        for metric_name in ['val_loss']:
            metric_total = 0
            for output in outputs:
                metric_value = output[metric_name]

                if self.trainer.use_dp or self.trainer.use_ddp2:
                    metric_value = torch.mean(metric_value)

            metric_total += metric_value
            tqdm_dict[metric_name] = metric_total.item() / len(outputs)

        return {"log": tqdm_dict, "val_loss": tqdm_dict["val_loss"]}

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr,
                          eps=self.hparams.adam_epsilon)
        t_total = (
                (len(self.train_dataset) // (self.hparams.train_batch_size * max(1, self.hparams.gpus)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup, num_training_steps=t_total
        )

        return [optimizer], [scheduler]

    def train_dataloader(self):
        data_loader = DataLoader(self.train_dataset,
                                 batch_size=self.hparams.train_batch_size,
                                 shuffle=True,
                                 num_workers=16)
        return data_loader

    def val_dataloader(self):
        data_loader = DataLoader(self.val_dataset,
                                 batch_size=self.hparams.eval_batch_size,
                                 num_workers=4)
        return data_loader

    def load_and_cache_data(self, data_type):
        # Load data features from cache or dataset file
        cached_file = os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}_{}".format(
                data_type,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.max_src_seq_length),
                str(self.hparams.max_tgt_seq_length),
            ),
        )
        if os.path.exists(cached_file) and not self.hparams.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_file)
            dataset = torch.load(cached_file)
        else:
            logger.info("Creating features from dataset file at %s", self.hparams.data_dir)
            dataset = MatchSumDataset(
                self.tokenizer, data_dir=self.hparams.data_dir, type_path=data_type,
                max_src_seq_length=self.hparams.max_src_seq_length,
                max_tgt_seq_length=self.hparams.max_tgt_seq_length
            )
            logger.info("Saving features into cached file %s", cached_file)
            torch.save(dataset, cached_file)

        return dataset

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
        )
        parser.add_argument(
            "--model_name",
            default="base_model_wo_optim",
            type=str,
        )
        parser.add_argument(
            "--vocab_name",
            default="wp_kt.model3_1",
            type=str,
        )
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
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the dataset files for the summarization task.",
        )

        parser.add_argument("--lr", default=1e-2, type=float, help="The initial learning rate for encoder.")
        parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup", default=3000, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument(
            "--num_train_epochs",
            default=30,
            type=int,
            help="Total number of training epochs to perform."
        )
        parser.add_argument("--train_batch_size", default=6, type=int)
        parser.add_argument("--eval_batch_size", default=1, type=int)

        return parser
