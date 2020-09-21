import logging
import os
from itertools import chain

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from src.common.dataset.dataset import SummarizationDataset
from src.common.tokenization.tokenization_kt import Tokenizer
from src.single_summary.bertabs import BertAbs

logger = logging.getLogger(__name__)


class Summarization(pl.LightningModule):
    def __init__(self, hparams):
        super(Summarization, self).__init__()
        self.hparams = hparams
        self.tokenizer = Tokenizer(os.path.join(self.hparams.model_name_or_path,
                                                self.hparams.vocab_name), max_len=512)
        self.model = BertAbs(self.hparams)

    def prepare_data(self):
        # load_and cache_data
        train_dataset = self.load_and_cache_data(data_type="train")
        val_dataset = self.load_and_cache_data(data_type="val")

        # assign to use in dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def forward(self, batch):
        outputs = self.model(encoder_input_ids=batch["source_ids"],
                             encoder_attention_mask=batch["source_mask"],
                             decoder_input_ids=batch["target_ids"],
                             decoder_attention_mask=None)

        # outputs = self.model(encoder_input_ids=batch["source_ids"],
        #                      token_type_ids=batch["source_type"],
        #                      encoder_attention_mask=batch["source_mask"],
        #                      decoder_input_ids=batch["target_ids"],
        #                      decoder_attention_mask=None)

        return self.model.generator(outputs)

    def is_logger(self):
        return self.trainer.proc_rank <= 0

    def nll_loss(self, pred, target, ignore_index=0):
        return F.nll_loss(pred, target, ignore_index=ignore_index)

    def training_step(self, batch, batch_idx, optimizer_idx):
        preds = self(batch)

        targets = batch["target_ids"][:, 1:].reshape(-1)
        preds = preds.view(-1, preds.size(2))
        loss = F.nll_loss(preds, targets, ignore_index=0)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        preds = self(batch)

        targets = batch["target_ids"][:, 1:].reshape(-1)
        preds = preds.view(-1, preds.size(2))
        loss = F.nll_loss(preds, targets, ignore_index=0)

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
        encoder_optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.bert.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        decoder_named_parameters = list(chain(self.model.decoder.named_parameters(),
                                              self.model.generator.named_parameters()))
        decoder_optimizer_grouped_parameters = [
            {
                "params": [p for n, p in decoder_named_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in decoder_named_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        encoder_optimizer = AdamW(encoder_optimizer_grouped_parameters,
                                  lr=self.hparams.encoder_lr,
                                  eps=self.hparams.adam_epsilon)
        decoder_optimizer = AdamW(decoder_optimizer_grouped_parameters,
                                  lr=self.hparams.decoder_lr,
                                  eps=self.hparams.adam_epsilon)

        t_total = (
                (len(self.train_dataset) // (self.hparams.train_batch_size * max(1, self.hparams.gpus)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        encoder_scheduler = get_linear_schedule_with_warmup(
            encoder_optimizer, num_warmup_steps=self.hparams.encoder_warmup, num_training_steps=t_total
        )
        decoder_scheduler = get_linear_schedule_with_warmup(
            decoder_optimizer, num_warmup_steps=self.hparams.decoder_warmup, num_training_steps=t_total
        )

        return [encoder_optimizer, decoder_optimizer], \
               [encoder_scheduler, decoder_scheduler]

    # def optimizer_step(self, epoch, batch_nb, optimizer, optimizer_idx, second_order_closure=None):
    #     if optimizer_idx == 0:
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         # self.encoder_scheduler.step()
    #
    #     if optimizer_idx == 1:
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         # self.decoder_scheduler.step()

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
            dataset = SummarizationDataset(
                self.tokenizer, data_dir=self.hparams.data_dir, type_path=data_type,
                max_src_seq_length=self.hparams.max_src_seq_length,
                max_tgt_seq_length=self.hparams.max_tgt_seq_length
            )
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
            help="The input data dir. Should contain the dataset files for the CNN/DM summarization task.",
        )
        parser.add_argument("--encoder_lr", default=5e-5, type=float, help="The initial learning rate for encoder.")
        parser.add_argument("--decoder_lr", default=0.01, type=float, help="The initial learning rate for decoder.")
        parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--encoder_warmup", default=5000, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--decoder_warmup", default=2000, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument(
            "--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform."
        )

        parser.add_argument("--share_emb", default=True)
        parser.add_argument("--train_batch_size", default=16, type=int)
        parser.add_argument("--eval_batch_size", default=2, type=int)

        return parser
