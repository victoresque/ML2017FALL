# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import json
import os
import argparse
import pickle

from os import path
from tqdm import tqdm
from unidecode import unidecode

from gensim.models.word2vec import Word2Vec
from keras.preprocessing.sequence import pad_sequences

import jieba
jieba.set_dictionary('dict/dict.txt.big')

def jieba_tokenizer():
    # TODO: change to jieba

    def tokenize_context(context):
        parsed = jieba.cut(context)
        tokens = []
        char_offsets = []
        offset = 0
        for w in parsed:
            tokens.append(w)
            char_offsets.append([offset, offset+len(w)-1])
            offset += len(w)

        return tokens, char_offsets

    return tokenize_context


def word2vec(word2vec_path):
    print('Reading word2vec data... ', end='')
    model = Word2Vec.load(word2vec_path).wv
    print('Done')

    def get_word_vector(word):
        try:
            return model[word]
        except KeyError:
            return np.zeros(model.vector_size)

    return get_word_vector


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word2vec_path', type=str,
                        default='word2vec/zh.bin',
                        help='Word2Vec vectors file path')
    parser.add_argument('--outfile', type=str, default='data/tmp.pkl',
                        help='Desired path to output pickle')
    parser.add_argument('--include_str', action='store_true',
                        help='Include strings')
    parser.add_argument('data', type=str, help='Data json')
    args = parser.parse_args()

    if not args.outfile.endswith('.pkl'):
        args.outfile += '.pkl'

    print('Reading json data... ', end='')
    with open(args.data, 'r', encoding='utf-8') as fd:
        samples = json.load(fd)
    print('Done!')

    tokenize = jieba_tokenizer()
    word_vector = word2vec(args.word2vec_path)

    def parse_sample(context, question, answer_start, answer_end, **kwargs):
        inputs = []
        targets = []

        tokens, char_offsets = tokenize(context)
        try:
            answer_start = [s <= answer_start < e
                            for s, e in char_offsets].index(True)
            targets.append(answer_start)
            answer_end   = [s <= answer_end < e
                            for s, e in char_offsets].index(True)
            targets.append(answer_end)
        except ValueError:
            return None

        tokens = [unidecode(token) for token in tokens]

        context_vecs = [word_vector(token) for token in tokens]
        context_vecs = np.vstack(context_vecs).astype(np.float32)
        inputs.append(context_vecs)

        if args.include_str:
            context_str = [np.fromstring(token, dtype=np.uint8).astype(np.int32)
                           for token in tokens]
            context_str = pad_sequences(context_str, maxlen=25)
            inputs.append(context_str)

        tokens, char_offsets = tokenize(question)
        tokens = [unidecode(token) for token in tokens]

        question_vecs = [word_vector(token) for token in tokens]
        question_vecs = np.vstack(question_vecs).astype(np.float32)
        inputs.append(question_vecs)

        if args.include_str:
            question_str = [np.fromstring(token, dtype=np.uint8).astype(np.int32)
                            for token in tokens]
            question_str = pad_sequences(question_str, maxlen=25)
            inputs.append(question_str)

        return [inputs, targets]

    print('Parsing samples... ', end='')
    samples = [parse_sample(**sample) for sample in tqdm(samples)]
    samples = [sample for sample in samples if sample is not None]
    print('Done!')

    # Transpose
    def transpose(x):
        return map(list, zip(*x))

    data = [transpose(input) for input in transpose(samples)]


    print('Writing to file {}... '.format(args.outfile), end='')
    with open(args.outfile, 'wb') as fd:
        pickle.dump(data, fd, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done!')
