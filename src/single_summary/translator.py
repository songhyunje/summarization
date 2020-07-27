import torch


def build_predictor(args, tokenizer, symbols, model):
    # we should be able to refactor the global scorer a lot
    scorer = GNMTGlobalScorer(args.alpha, length_penalty="wu")
    translator = Translator(args, model, tokenizer, symbols, global_scorer=scorer)
    return translator


class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`
    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
    """

    def __init__(self, alpha, length_penalty):
        self.alpha = alpha
        penalty_builder = PenaltyBuilder(length_penalty)
        self.length_penalty = penalty_builder.length_penalty()

    def score(self, beam, logprobs):
        """
        Rescores a prediction based on penalty functions
        """
        normalized_probs = self.length_penalty(beam, logprobs, self.alpha)
        return normalized_probs


class PenaltyBuilder(object):
    """
    Returns the Length and Coverage Penalty function for Beam Search.
    Args:
        length_pen (str): option name of length pen
        cov_pen (str): option name of cov pen
    """

    def __init__(self, length_pen):
        self.length_pen = length_pen

    def length_penalty(self):
        if self.length_pen == "wu":
            return self.length_wu
        elif self.length_pen == "avg":
            return self.length_average
        else:
            return self.length_none

    """
    Below are all the different penalty terms implemented so far
    """

    def length_wu(self, beam, logprobs, alpha=0.0):
        """
        NMT length re-ranking score from
        "Google's Neural Machine Translation System" :cite:`wu2016google`.
        """

        modifier = ((5 + len(beam.next_ys)) ** alpha) / ((5 + 1) ** alpha)
        return logprobs / modifier

    def length_average(self, beam, logprobs, alpha=0.0):
        """
        Returns the average probability of tokens in a sequence.
        """
        return logprobs / len(beam.next_ys)

    def length_none(self, beam, logprobs, alpha=0.0, beta=0.0):
        """
        Returns unmodified scores.
        """
        return logprobs


class Translator(object):
    """
    Uses a model to translate a batch of sentences.
    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       beam_trace (bool): trace beam search for debugging
    """

    def __init__(self, args, model, vocab, symbols, global_scorer=None):
        self.args = args
        self.model = model
        self.generator = self.model.generator
        self.vocab = vocab
        self.symbols = symbols
        self.start_token = symbols["BOS"]
        self.end_token = symbols["EOS"]
        self.period_token = symbols["PERIOD"]

        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length

    def translate(self, batch, attn_debug=False):
        """ Generates summaries from one batch of data.
        """
        self.model.eval()
        with torch.no_grad():
            batch_data = self.translate_batch(batch)
            translations = self.from_batch(batch_data)
        return translations

    def translate_batch(self, batch, fast=False):
        """
        Translate a batch of sentences.
        Mostly a wrapper around :obj:`Beam`.
        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)
        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            return self._fast_translate_batch(batch, self.max_length, min_length=self.min_length)

    # Where the beam search lives
    # I have no idea why it is being called from the method above
    def _fast_translate_batch(self, batch, max_length, min_length=0):
        """ Beam Search using the encoder inputs contained in `batch`.
        """

        beam_size = self.beam_size
        # batch_size = batch.batch_size
        batch_size = 1

        # src = batch.src
        # segs = batch.segs
        # mask_src = batch.mask_src

        src = batch["input_ids"]
        mask_src = batch["attention_mask"]

        # if self.args.encoder != 'kt':
        #     segs = batch["token_type_ids"]
        #     src_features = self.model.bert(src, segs, mask_src)
        # else:
        #     src_features = self.model.bert(src, mask_src)

        src_features = self.model.bert(src, mask_src)
        dec_states = self.model.decoder.init_decoder_state(src, src_features, with_cache=True)
        device = src_features[0].device

        # encoder_hidden_states = encoder_output[0]
        # dec_state = self.decoder.init_decoder_state(encoder_input_ids, encoder_hidden_states)

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(lambda state, dim: tile(state, beam_size, dim=dim))
        src_features = tile(src_features[0], beam_size, dim=0)
        batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(0, batch_size * beam_size, step=beam_size, dtype=torch.long, device=device)
        alive_seq = torch.full([batch_size * beam_size, 1], self.start_token, dtype=torch.long, device=device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = torch.tensor([0.0] + [float("-inf")] * (beam_size - 1), device=device).repeat(batch_size)

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch

        for step in range(max_length):
            decoder_input = alive_seq[:, -1].view(1, -1)

            # Decoder forward.
            decoder_input = decoder_input.transpose(0, 1)

            dec_out, dec_states = self.model.decoder(decoder_input, src_features, dec_states, step=step)

            # Generator forward.
            log_probs = self.generator(dec_out.transpose(0, 1).squeeze(0))
            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, self.end_token] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            if self.args.block_trigram:
                cur_len = alive_seq.size(1)
                if cur_len > 3:
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        # KT
                        words = [self.vocab.I2T[w] for w in words]

                        # ETRI
                        # words = [self.vocab.ids_to_tokens[w] for w in words]

                        # words = " ".join(words).replace(" ##", "").split()
                        words = ' '.join(words).replace(u'▁', '').split()
                        if len(words) <= 3:
                            continue
                        trigrams = [(words[i - 1], words[i], words[i + 1]) for i in range(1, len(words) - 1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = -10e20

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = topk_beam_index + beam_offset[: topk_beam_index.size(0)].unsqueeze(1)
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat([alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1)

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)

            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((topk_scores[i, j], predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]

                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished).view(-1, alive_seq.size(-1))
            # Reorder states.
            select_indices = batch_index.view(-1)
            src_features = src_features.index_select(0, select_indices)
            dec_states.map_batch_fn(lambda state, dim: state.index_select(dim, select_indices))

        return results

    def from_batch(self, translation_batch):
        # batch = translation_batch["batch"]
        assert len(translation_batch["gold_score"]) == len(translation_batch["predictions"])
        # batch_size = batch.batch_size
        batch_size = 1

        # preds, _, _, tgt_str, src = (
        #     translation_batch["predictions"],
        #     translation_batch["scores"],
        #     translation_batch["gold_score"],
        #     batch.tgt_str,
        #     batch.src,
        # )
        preds, _, _ = (
            translation_batch["predictions"],
            translation_batch["scores"],
            translation_batch["gold_score"],
        )

        translations = []
        for b in range(batch_size):
            # pred_sents = [int(n) for n in preds[b][0]]

            pred_sents = []
            for n in preds[b][0]:
                n = int(n)
                if n == self.end_token:
                    break
                pred_sents.append(n)

            if pred_sents[-1] != self.period_token:
                if self.period_token in pred_sents[::-1]:
                    pred_sents = pred_sents[:len(pred_sents) - pred_sents[::-1].index(self.period_token)]
                else:
                    pred_sents += [self.period_token]

            translations.append(pred_sents)

        # for b in range(batch_size):
        #     pred_sents = self.vocab.convert_ids_to_tokens([int(n) for n in preds[b][0]])
        #     pred_sents = " ".join(pred_sents).replace(" ##", "")
        #     # gold_sent = " ".join(tgt_str[b].split())
        #     # raw_src = [self.vocab.ids_to_tokens[int(t)] for t in src[b]][:500]
        #     # raw_src = " ".join(raw_src)
        #     # translation = (pred_sents, gold_sent, raw_src)
        #     translation = pred_sents
        #     translations.append(translation)

        return translations

def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1).transpose(0, 1).repeat(count, 1).transpose(0, 1).contiguous().view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x
