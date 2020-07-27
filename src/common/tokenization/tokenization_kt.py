# -*- coding: utf-8 -*-

import logging
import os
from itertools import chain
from unicodedata import normalize

import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_vocab(vocab_file, encode='utf8'):
    toks = normalize('NFD', str(os.popen('cat %s' % vocab_file).read())).splitlines()
    return toks, {tok: i for i, tok in enumerate(toks)}


# def word_tokenize(word, vocab):
#    wordlen = len(word)
#    start = 0
#    end = len(word)
#    toks = []
#    while start < wordlen and end > 0 and start != end:
#        if word[start:end] in vocab:
#            toks.append(word[start:end])
#            start = end
#            end = wordlen
#        else:
#            end -= 1
#    return toks

def word_tokenize(word, vocab):
    wordlen = len(word)
    start = 0
    end = len(word)
    toks = []
    ftok = '<!#unk>'
    while start < wordlen:
        if start == end:
            ftok = '<!#unk>'
            start += 1
            end = wordlen
        if word[start:end] in vocab:
            ftok = word[start:end]
            start = end
            end = wordlen
            break
        else:
            end -= 1

    word = word[start:]
    start = 0
    end = len(word)

    while end > 0:
        if start == end:
            toks.append('<!#unk>')
            end -= 1
            start = 0
        if word[start:end] in vocab:
            toks.append(word[start:end])
            end = start
            start = 0
        else:
            start += 1
    toks.append(ftok)
    return reversed(toks)


#
class Tokenizer(object):
    def __init__(self, vocab_file, max_len=1e10, encode='utf8'):
        self.I2T, self.T2I = load_vocab(vocab_file)
        self.vocab = self.T2I
        self.vocab_size = len(self.I2T)
        self.max_len = max_len
        self.encode = encode
        self.padtok = '<!#pad>'
        self.masktok = '<!#mask>'
        self.unktok = '<!#unk>'
        self.bgntok = '<!#s>'
        self.endtok = '<!#/s>'

    def tokenize(self, sent, be_tok=False):
        # if isinstance(sent, unicode):
        #     sent = sent
        # else:
        #     sent = unicode(sent, self.encode)

        if be_tok:
            sent = u'<!#s> ▁%s <!#/s>' % (normalize('NFD', sent.upper()).replace(' ', u' ▁'))
        else:
            sent = u'▁%s' % (normalize('NFD', sent.upper()).replace(' ', u' ▁'))
        words = sent.split(' ')

        return list(chain(*[word_tokenize(word, self.T2I) for word in words]))

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab["<!#unk>"])
        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def encode_plus(self, text, max_length=None, add_special_tokens=True, pad_to_max_length=False, return_tensors=None):
        if isinstance(text, str):
            tokens = self.tokenize(text, be_tok=add_special_tokens)
        else:
            tokens = text
            # check that the tokens starts with special token
            if add_special_tokens and tokens[0] != self.bgntok:
                tokens = [self.bgntok] + tokens + [self.endtok]

        encoded_inputs = {}
        encoded_inputs['input_ids'] = [self.T2I[t] if t in self.T2I else self.T2I[self.unktok] for t in tokens]
        if max_length and len(encoded_inputs["input_ids"]) > max_length:
            encoded_inputs["input_ids"] = encoded_inputs["input_ids"][:max_length]

        if pad_to_max_length and max_length is None and self.max_len > 10000:
            logger.warning(
                "Sequence can't be padded as no maximum length is specified and the model maximum length is too high."
            )

        needs_to_be_padded = pad_to_max_length and (
                max_length
                and len(encoded_inputs["input_ids"]) < max_length
                or max_length is None
                and len(encoded_inputs["input_ids"]) < self.max_len <= 10000
        )

        if needs_to_be_padded:
            difference = (max_length if max_length is not None else self.max_len) - len(encoded_inputs["input_ids"])
            encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"]) + [0] * difference
            encoded_inputs["input_ids"] = encoded_inputs["input_ids"] + [self.T2I[self.padtok]] * difference
        else:
            encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"])

        if return_tensors == "pt":
            encoded_inputs["input_ids"] = torch.tensor([encoded_inputs["input_ids"]])
            encoded_inputs["attention_mask"] = torch.tensor([encoded_inputs["attention_mask"]])

        return encoded_inputs

    def encode_plus_mds(self, list_of_text,
                        max_sent_length=None,
                        max_length=None,
                        pad_to_max_length=False,
                        return_tensors=None):
        tokens = []
        for text in list_of_text:
            # tokenized = self.tokenize(text)
            # if max_sent_length:
            #     tokenized = tokenized[:max_sent_length - 1] + ['.']  # ends with period
            # tokens += ['<!#s>'] + tokenized + ['<!#/s>']

            tokenized = self.tokenize(text, True)
            if max_sent_length:
                tokenized = tokenized[:max_sent_length - 1] + ['<!#/s>']
            tokens += tokenized

        # tokens += ['<!#s>'] + tokens + ['<!#/s>']
        encoded_inputs = {}
        encoded_inputs['input_ids'] = [self.T2I[t] if t in self.T2I else self.T2I[self.unktok] for t in tokens]
        if max_length and len(encoded_inputs["input_ids"]) > max_length:
            encoded_inputs["input_ids"] = encoded_inputs["input_ids"][:max_length - 1] + [self.T2I[self.endtok]]

        if pad_to_max_length and max_length is None and self.max_len > 10000:
            logger.warning(
                "Sequence can't be padded as no maximum length is specified and the model maximum length is too high."
            )

        needs_to_be_padded = pad_to_max_length and (
                max_length
                and len(encoded_inputs["input_ids"]) < max_length
                or max_length is None
                and len(encoded_inputs["input_ids"]) < self.max_len <= 10000
        )

        if needs_to_be_padded:
            difference = (max_length if max_length is not None else self.max_len) - len(encoded_inputs["input_ids"])
            encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"]) + [0] * difference
            encoded_inputs["input_ids"] = encoded_inputs["input_ids"] + [self.T2I[self.padtok]] * difference
        else:
            encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"])

        if return_tensors == "pt":
            encoded_inputs["input_ids"] = torch.tensor([encoded_inputs["input_ids"]])
            encoded_inputs["attention_mask"] = torch.tensor([encoded_inputs["attention_mask"]])

        return encoded_inputs

    def decode(self, token_ids):
        tmp = []
        for index in token_ids:
            if index in [0, 1, 2]:  # pad, mask, unk
                continue
            tmp.append(self.I2T[index])
        tokens = normalize('NFC', ''.join(tmp))
        return tokens.replace(u'▁', ' ').strip()

    def starts_with_space(self, token_id):
        token = normalize('NFC', self.I2T[token_id])
        return token.startswith(u'▁')

    def convert_id(self, token_id):
        token = u'▁' + self.I2T[token_id]
        return self.T2I.get(token, token_id)   # if we fail, return original token id

    def S2WIs(self, sent, keep=1.):
        toks = self.tokenize(sent)
        if keep < 1.:
            toks = [tok if np.random.random() < keep else self.unktok for tok in toks]
        return [self.T2I[t] if t in self.T2I else self.T2I[self.unktok] for t in toks]

    def WIs2S(self, WIs):
        return normalize('NFC', ' '.join([self.I2T[i] for i in WIs])).replace(u'▁', '')


def pad_sequences(batch, maxlen=None, dtype='int32', trunc='pre', value=0):
    batch_num = len(batch)
    bmaxslen = maxlen if maxlen else len(max(batch, key=len))

    output = np.full((batch_num, bmaxslen), value).astype(np.int32)

    for sidx, sent in enumerate(batch):
        if trunc == 'pre':
            truncated = sent[max(len(sent) - bmaxslen, 0):]
        else:
            truncated = sent[:bmaxslen]
        output[sidx, :len(truncated)] = truncated

    return output


if __name__ == '__main__':
    tokenizer = Tokenizer('../../resources/bert-kt-large/wp_kt.model3_1', max_len=128)

    # s3 = u'우리카드에 역전승을 거뒀다.'
    # s3_tokens = tokenizer.tokenize(s3)
    # s3_ids = tokenizer.S2WIs(s3)
    # s3_ = tokenizer.WIs2S(s3_ids)
    #
    # s3_ids[4] = tokenizer.convert_id(s3_ids[4])
    # s3__ = tokenizer.decode(s3_ids[:3] + s3_ids[4:])
    # print(s3)
    # print(s3_tokens)
    # print(s3_ids)
    # print(s3_)
    # print(s3__)

    s3 = u'가성비가 좋다.'
    s3_tokens = tokenizer.tokenize(s3)
    s3_ids = tokenizer.S2WIs(s3)
    s3_ = tokenizer.WIs2S(s3_ids)
    print(s3)
    print(s3_tokens)
    print(s3_ids)
    print(s3_)

    # s1 = u'문재인 대통령께서 입장하십니다.'
    # s1_tokens = tokenizer.tokenize(s1)
    # s1_ids = tokenizer.S2WIs(s1)
    # s1_ = tokenizer.WIs2S(s1_ids)
    # print(s1)
    # print(s1_tokens)
    # print(s1_ids)
    # print(s1_)
    #
    # s2 = u'잼는 영화 좀 틀어줘봐'
    # s2_tokens = tokenizer.tokenize(s2)
    # s2_ids = tokenizer.S2WIs(s2)
    # s2_ = tokenizer.WIs2S(s2_ids)
    # print(s2)
    # print(s2_tokens)
    # print(s2_ids)
    # print(s2_)
    #
    # s3 = u'임진왜란은 1592년에 발발했고 6년간 지속되었다'
    # s3_tokens = tokenizer.tokenize(s3)
    # s3_ids = tokenizer.S2WIs(s3)
    # s3_ = tokenizer.WIs2S(s3_ids)
    # print(s3)
    # print(s3_tokens)
    # print(s3_ids)
    # print(s3_)
    #
    # s3 = u'공자는 중국 고대의 사상가이죠'
    # s3_tokens = tokenizer.tokenize(s3)
    # s3_ids = tokenizer.S2WIs(s3)
    # s3_ = tokenizer.WIs2S(s3_ids)
    # print(s3)
    # print(s3_tokens)
    # print(s3_ids)
    # print(s3_)
    #
    # s3 = u'누군지 잘 모르겠네요'
    # s3_tokens = tokenizer.tokenize(s3)
    # s3_ids = tokenizer.S2WIs(s3)
    # s3_ = tokenizer.WIs2S(s3_ids)
    # print(s3)
    # print(s3_tokens)
    # print(s3_ids)
    # print(s3_)
    #
    # s3 = u'말하기가 제일 쉬웠어요'
    # s3_tokens = tokenizer.tokenize(s3)
    # s3_ids = tokenizer.S2WIs(s3)
    # s3_ = tokenizer.WIs2S(s3_ids)
    # print(s3)
    # print(s3_tokens)
    # print(s3_ids)
    # print(s3_)
    #
    # s3 = u'화성초등학교'
    # s3_tokens = tokenizer.tokenize(s3)
    # s3_ids = tokenizer.S2WIs(s3)
    # s3_ = tokenizer.WIs2S(s3_ids)
    # print(s3)
    # print(s3_tokens)
    # print(s3_ids)
    # print(s3_)

