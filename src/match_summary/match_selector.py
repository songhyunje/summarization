import torch


def build_selector(args, tokenizer, symbols, model):
    selector = Selector(args, model, tokenizer, symbols)
    return selector


class Selector(object):
    def __init__(self, args, model, vocab, symbols):
        self.args = args
        self.model = model
        self.vocab = vocab
        self.symbols = symbols
        self.start_token = symbols["BOS"]
        self.end_token = symbols["EOS"]
        self.period_token = symbols["PERIOD"]

        self.min_length = args.min_length
        self.max_length = args.max_length

    def select(self, batch, max_length=None, min_length=None):
        if min_length is not None:
            self.min_length = min_length
        if max_length is not None:
            self.max_length = max_length

        self.model.eval()
        with torch.no_grad():
            sents = self.select_batch(batch)
        return sents

    def select_batch(self, batch):
        src_embeddings = self.model(input_ids=batch["source"]["input_ids"],
                                    attention_mask=batch["source"]["attention_mask"],
                                    output_all_encoded_layers=False)
        cands_embeddings = self.model(input_ids=batch["cands"]["input_ids"],
                                      attention_mask=batch["cands"]["attention_mask"],
                                      output_all_encoded_layers=False)

        doc_emb = src_embeddings[:, 0, :]
        candidate_emb = cands_embeddings[:, 0, :]
        score = torch.cosine_similarity(candidate_emb, doc_emb, dim=-1)
        _, predicted = torch.max(score.data, 0)
        return batch["cands"]["sents"][predicted]

    def select_batches(self, batch):
        srcs_embeddings = self.model(input_ids=batch["srcs"]["input_ids"],
                                     attention_mask=batch["srcs"]["attention_mask"],
                                     output_all_encoded_layers=False)
        cands_embeddings = self.model(input_ids=batch["cands"]["input_ids"],
                                      attention_mask=batch["cands"]["attention_mask"],
                                      output_all_encoded_layers=False)

        srcs_emb = srcs_embeddings[:, 0, :].unsqueeze(0)
        cands_emb = cands_embeddings[:, 0, :].unsqueeze(1)
        score = torch.cosine_similarity(cands_emb, srcs_emb, dim=-1)
        _, predicted = torch.max(score.detach().sum(1).data, 0)
        return batch["cands"]["sents"][predicted]
