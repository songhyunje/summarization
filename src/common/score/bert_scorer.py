import logging
import os

import torch

from src.common.model.modeling import BertModel
from src.common.tokenization.tokenization_kt import Tokenizer

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, 'INFO'))
logger = logging.getLogger(__name__)


class BertScorer(object):
    def __init__(self, model_name_or_path="../resources/bert-kt-large/", device='cuda:0', batch_size=16):
        self.tokenizer = Tokenizer(os.path.join(model_name_or_path, 'wp_kt.model3_1'), max_len=512)
        model_state_dict = torch.load(os.path.join(model_name_or_path, 'base_model_wo_optim'))
        self.model = BertModel.from_pretrained(model_name_or_path, **model_state_dict)
        self.model.eval()
        self.device = device
        self.model.to(self.device)
        self.batch_size = batch_size
        self.hyps_cached = dict()

    def get_bert_embeddings(self, sents, max_len):
        encoded_ids, encoded_attention_masks, tokens_list = [], [], []
        for sent in sents:
            tokens = self.tokenizer.tokenize(sent)
            encoded = self.tokenizer.encode_plus(tokens,
                                                 add_special_tokens=True,
                                                 max_length=max_len,
                                                 pad_to_max_length=True,
                                                 return_tensors="pt")

            for k, v in encoded.items():
                encoded[k] = v.to(self.device)

            tokens_list.append(tokens)
            encoded_ids.append(encoded['input_ids'].squeeze())
            encoded_attention_masks.append(encoded['attention_mask'].squeeze())

        with torch.no_grad():
            outputs = self.model(input_ids=torch.stack(encoded_ids),
                                 attention_mask=torch.stack(encoded_attention_masks))[0]

        return outputs, tokens_list, torch.stack(encoded_attention_masks)

    def greedy_cos_idf(self, refs, hyps, idf_weight, idf_dict_ref, idf_dict_hyp, max_len):
        ref_embeddings, ref_tokens, ref_masks = self.get_bert_embeddings(refs, max_len=max_len)
        # hyp_embeddings, hyp_tokens, hyp_masks = self.get_bert_embeddings(hyps, max_len=max_len)
        hyps_key = tuple(hyps)
        if hyps_key in self.hyps_cached:
            hyp_embeddings, hyp_tokens, hyp_masks = self.hyps_cached[hyps_key]
        else:
            hyp_embeddings, hyp_tokens, hyp_masks = self.get_bert_embeddings(hyps, max_len=max_len)
            self.hyps_cached = dict()
            self.hyps_cached[hyps_key] = (hyp_embeddings, hyp_tokens, hyp_masks)

        # normalized
        ref_embeddings.div_(torch.norm(ref_embeddings, dim=-1).unsqueeze(-1))
        hyp_embeddings.div_(torch.norm(hyp_embeddings, dim=-1).unsqueeze(-1))

        batch_size = ref_embeddings.size(0)
        sim = torch.bmm(ref_embeddings, hyp_embeddings.transpose(1, 2))
        masks = torch.bmm(ref_masks.unsqueeze(2).float(), hyp_masks.unsqueeze(1).float()).float()
        masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)
        sim = sim * masks

        precision = sim.max(dim=2)[0]
        recall = sim.max(dim=1)[0]

        if idf_weight and idf_dict_ref and idf_dict_hyp:
            hyp_idf = []
            for tokens in hyp_tokens:
                hyp_idf.append([idf_dict_hyp.get(token, 0.0) for token in tokens])
            hyp_idf = torch.FloatTensor(hyp_idf).to(precision.device)
            hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))

            ref_idf = []
            for tokens in ref_tokens:
                ref_idf.append([idf_dict_ref.get(token, 0.0) for token in tokens])
            ref_idf = torch.FloatTensor(ref_idf).to(recall.device)
            ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))

            P = (precision * hyp_idf).sum(dim=1)
            R = (recall * hyp_idf).sum(dim=1)
        else:
            hyp_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=precision.device)
            ref_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=recall.device)

            hlen = [len(tokens) for tokens in hyp_tokens]
            for i, hl in enumerate(hlen):
                hyp_mask[i, 1:hl + 1] = True

            rlen = [len(tokens) for tokens in ref_tokens]
            for i, rl in enumerate(rlen):
                ref_mask[i, 1:rl + 1] = True

            precision.masked_fill_(~hyp_mask, 0)
            recall.masked_fill_(~ref_mask, 0)

            P = precision.sum(dim=1) / hyp_mask.sum(dim=1)
            R = recall.sum(dim=1) / ref_mask.sum(dim=1)

        F1 = 2 * P * R / (P + R)
        return F1

    def score(self, refs, hyps, idf_dict_ref=None, idf_dict_hyp=None,
              idf_weight=True, max_len=512, rescale_with_baseline=False):
        scores = []
        for batch in range(0, len(refs), self.batch_size):
            batch_refs = refs[batch: batch + self.batch_size]
            batch_hyps = hyps[batch: batch + self.batch_size]
            F1 = self.greedy_cos_idf(batch_refs, batch_hyps,
                                     idf_weight=idf_weight,
                                     idf_dict_ref=idf_dict_ref,
                                     idf_dict_hyp=idf_dict_hyp,
                                     max_len=max_len)
            scores.extend(F1.cpu().numpy())
        return scores
