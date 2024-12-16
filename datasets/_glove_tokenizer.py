import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, normalizers
from transformers import BertTokenizer


def build_bert_vocab():
    return bertTokenizer()

def build_glove_vocab(glove_fn):
    words, vectors = load_glove_embedding(glove_fn)
    words = ['[PAD]', '[UNK]'] + words
    zeros = np.zeros_like(vectors[0])
    vectors = [zeros, zeros] + vectors

    vocab = {w: idx for idx, w in enumerate(words)}
    embeddings = np.stack(vectors)
    return GloveTokenizer(vocab, unk_token='[UNK]', embeddings_ori=embeddings)


def load_glove_embedding(glove_fn):
    words = []
    vectors = []
    with open(glove_fn, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line != "":
                space_idx = line.find(' ')
                word = line[:space_idx]
                words.append(word)

                numbers = line[space_idx + 1:]
                float_numbers = [float(number_str) for number_str in numbers.split()]
                vector = np.array(float_numbers)
                vectors.append(vector)
    return words, vectors


class GloveTokenizer:
    def __init__(self, vocab, unk_token, embeddings_ori):
        self.vocab = vocab
        self.unk_token = unk_token
        self.unk_token_id = self.vocab[self.unk_token]
        self.embeddings_ori = torch.tensor(embeddings_ori,dtype=torch.float)

        self.tk = Tokenizer(models.WordLevel(vocab, unk_token=unk_token))
        self.tk.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.StripAccents()])
        self.tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.Punctuation()])

    def get_vocab(self):
        return self.vocab

    def __call__(self, sentence_or_words,
                 is_split_into_words=True,
                 max_length=None,
                 padding='max_length',
                 truncation=True,
                 return_tensors='pt'):
        self.tk.enable_padding(length=max_length)
        self.tk.enable_truncation(max_length=max_length)
        return self.tk.encode(sentence_or_words, is_pretokenized=is_split_into_words)

class bertTokenizer:
    def __init__(self, bert_name='bert-base-uncased'):
        self.tk = BertTokenizer.from_pretrained(bert_name)
        self.vocab = self.tk.get_vocab()
        self.unk_token = self.tk.unk_token
        self.unk_token_id = self.vocab[self.unk_token]
    def get_vocab(self):
        return self.vocab

    def __call__(self, sentence_or_words,
                 is_split_into_words=True,
                 max_length=None,
                 padding='max_length',
                 truncation=True,
                 return_tensors='pt'):
        return self.tk.encode_plus(
            sentence_or_words,
            max_length=10,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

if __name__ == '__main__':
    pass