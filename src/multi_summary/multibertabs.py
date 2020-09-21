import copy
import torch
import torch.nn as nn
import os

from torch.nn.init import xavier_uniform_

from src.multi_summary.decoder import TransformerDecoder
from src.multi_summary.configuration_multibertabs import DecoderConfig
from src.common.model.modeling import BertModel


class MultiBertAbs(nn.Module):
    def __init__(self, hparams, checkpoint=None):
        super().__init__()
        self.hparams = hparams
        self.decoder_args = DecoderConfig()

        if self.hparams.load_from_single_model:
            single_summary_model = torch.load(self.hparams.load_from_single_model)
            state_dict = single_summary_model.get('state_dict', {})

            bert_state_dict = {}
            for key in list(state_dict):
                if key.startswith("model.bert"):
                    bert_state_dict[key[6:]] = state_dict.pop(key)

            bert_model = {"state_dict": bert_state_dict}
            self.bert = BertModel.from_pretrained(self.hparams.model_name_or_path, **bert_model)
            if self.hparams.encoder_freeze:
                for param in self.bert.parameters():
                    param.requires_grad = False
        else:
            model_state_dict = torch.load(os.path.join(self.hparams.model_name_or_path,
                                                       self.hparams.model_name))
            self.bert = BertModel.from_pretrained(self.hparams.model_name_or_path, **model_state_dict)
        self.vocab_size = self.bert.config.vocab_size

        # if self.hparams.max_src_seq_length > 512:
        #     my_pos_embeddings = nn.Embedding(self.hparams.max_src_seq_length, self.bert.config.hidden_size)
        #     my_pos_embeddings.weight.data[:512] = self.bert.embeddings.position_embeddings.weight.data
        #     my_pos_embeddings.weight.data[512:] = self.bert.embeddings.position_embeddings.weight.data[-1][
        #         None, :
        #     ].repeat(self.hparams.max_src_seq_length - 512, 1)
        #     self.bert.embeddings.position_embeddings = my_pos_embeddings

        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.config.hidden_size, padding_idx=0)

        if self.hparams.share_emb:
            tgt_embeddings.weight = copy.deepcopy(self.bert.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            self.decoder_args.dec_layers,
            self.decoder_args.dec_hidden_size,
            heads=self.decoder_args.dec_heads,
            d_ff=self.decoder_args.dec_ff_size,
            dropout=self.decoder_args.dec_dropout,
            embeddings=tgt_embeddings,
            vocab_size=self.vocab_size,
        )

        gen_func = nn.LogSoftmax(dim=-1)
        self.generator = nn.Sequential(nn.Linear(self.decoder_args.dec_hidden_size, self.vocab_size), gen_func)
        self.generator[0].weight = self.decoder.embeddings.weight

        if self.hparams.load_from_single_model:
            decoder_state_dict, generator_state_dict = {}, {}
            for key in list(state_dict):
                if key.startswith("model.decoder"):
                    decoder_state_dict[key[14:]] = state_dict.pop(key)
                elif key.startswith("model.generator"):
                    generator_state_dict[key[16:]] = state_dict.pop(key)

            self.decoder.load_state_dict(decoder_state_dict)
            self.generator.load_state_dict(generator_state_dict)
        else:
            self.init_weights()

        # load_from_checkpoints = False if checkpoint is None else True
        # if load_from_checkpoints:
        #     self.load_state_dict(checkpoint)

    def init_weights(self):
        for module in self.decoder.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        for p in self.generator.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            else:
                p.data.zero_()

    def forward(self, sources, target):
        encoder_input_ids = []
        encoder_hidden_states = []
        for src in sources:
            encoder_output = self.bert(
                input_ids=src["input_ids"], attention_mask=src["attention_mask"],
            )
            encoder_input_ids.append(src["input_ids"])
            encoder_hidden_states.append(encoder_output[-1])

        # encode_input_ids: batch x concat(source_seq)
        encoder_input_ids = torch.cat(encoder_input_ids, dim=1)
        # encoder_hidden_states: batch x concat(source_seq) x 768
        encoder_hidden_states = torch.cat(encoder_hidden_states, dim=1)
        # encoder_hidden_states = encoder_hidden_states[-1]

        dec_state = self.decoder.init_decoder_state(encoder_input_ids, encoder_hidden_states)

        decoder_input_ids = target["input_ids"]
        decoder_outputs, _ = self.decoder(decoder_input_ids[:, :-1], encoder_hidden_states, dec_state)

        return decoder_outputs

