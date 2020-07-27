import copy
import torch
import torch.nn as nn
import os

from torch.nn.init import xavier_uniform_

from src.common.model.modeling import BertModel
from src.single_summary.configuration_bertabs import DecoderConfig
from src.single_summary.decoder import TransformerDecoder


class BertAbs(nn.Module):
    def __init__(self, hparams, checkpoint=None):
        super().__init__()
        self.hparams = hparams
        self.decoder_args = DecoderConfig()

        model_state_dict = torch.load(os.path.join(self.hparams.model_name_or_path,
                                                   self.hparams.model_name))
        self.bert = BertModel.from_pretrained(self.hparams.model_name_or_path, **model_state_dict)
        self.vocab_size = self.bert.config.vocab_size
        # if self.encoder_args.max_position_embeddings > 512:
        #     my_pos_embeddings = nn.Embedding(self.encoder_args.max_position_embeddings, self.bert.config.hidden_size)
        #     my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
        #     my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][
        #         None, :
        #     ].repeat(self.encoder_args.max_position_embeddings - 512, 1)
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

        load_from_checkpoints = False if checkpoint is None else True
        if load_from_checkpoints:
            self.load_state_dict(checkpoint)

        self.init_weights()

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

    def forward(
        self, encoder_input_ids, decoder_input_ids, encoder_attention_mask, decoder_attention_mask,
    ):
        encoder_output = self.bert(
            input_ids=encoder_input_ids, attention_mask=encoder_attention_mask,
        )
        encoder_hidden_states = encoder_output[-1]  # batch x source_seq x 768
        dec_state = self.decoder.init_decoder_state(encoder_input_ids, encoder_hidden_states)
        decoder_outputs, _ = self.decoder(decoder_input_ids[:, :-1], encoder_hidden_states, dec_state)

        return decoder_outputs

    # def forward(
    #     self, encoder_input_ids, decoder_input_ids, token_type_ids,
    #         encoder_attention_mask, decoder_attention_mask,
    # ):
    #     encoder_output = self.bert(
    #         input_ids=encoder_input_ids, token_type_ids=token_type_ids, attention_mask=encoder_attention_mask,
    #     )
    #     encoder_hidden_states = encoder_output[0]
    #     dec_state = self.decoder.init_decoder_state(encoder_input_ids, encoder_hidden_states)
    #     decoder_outputs, _ = self.decoder(decoder_input_ids[:, :-1], encoder_hidden_states, dec_state)
    #
    #     return decoder_outputs
