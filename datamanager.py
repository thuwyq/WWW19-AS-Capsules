# coding: utf-8
#
# Copyright 2018 Yequan Wang
# Author: Yequan Wang (tshwangyequan@gmail.com)
#

import numpy as np
import re
import nltk


def cleanText(text):
    def add_space(matched):
        s = matched.group()
        return ' '+ s[0] + ' ' + s[-1]
    
    con_cleaned = re.sub(r'[^a-zA-Z0-9_\-\.,;:!?/\']', " ", text)
    con_cleaned = re.sub(r'[\.,;:!?/]+[a-zA-Z]', add_space, con_cleaned)
    
    try:
        wordtoken = nltk.word_tokenize(con_cleaned)
    except:
        print(con_cleaned)
        print(text)
        exit()
    content_tackled = ' '.join(wordtoken)

    def add_space_pre(matched):
        '''
        If word like "china." occured, split "china" and ".". 
        '''
        s = matched.group()
        return s[0] + ' ' + s[-1]
    content_tackled = re.sub(r'[a-zA-Z][\.,;:!?/]+', add_space_pre, content_tackled)
    
    return content_tackled


class DataManager(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

    def load_data(self, path, fname):
        with open('%s/%s' % (path, fname)) as f:
            lines = [eval(line.strip()) for line in f.readlines()]
        data = []
        dict_asp = {'food': 0, 'price': 1, 'service': 2, 'ambience': 3, 'anecdotes/miscellaneous': 4}
        for line in lines:
            label_asp = np.zeros(5, dtype='int64')
            label_sent = -1 * np.ones(5, dtype='int64')
            for tmp, tmp_sent in zip(line['aspects'], line['labels']):
                label_asp[dict_asp[tmp]] = 1
                label_sent[dict_asp[tmp]] = tmp_sent
            oneLine = {'sentence': cleanText(line['sentence']).lower().split(), 'aspects': label_asp, 'labels': label_sent}
            if oneLine != None:
                data.append(oneLine)
        return data

    def build_vocab(self, path, data, vocab=dict()):
        print("Creating vocabulary...")
        for inst in data:
            for token in inst['sentence']:    
                vocab[token] = (vocab[token] + 1) if (token in vocab) else 1
        vocab_list = sorted(vocab, key=vocab.get, reverse=True)
        vocab_list = vocab_list[: min(len(vocab_list), self.FLAGS.voc_size)]
        if ('<unk>' in vocab) == False:
            vocab_list.append('<unk>')

        vocab_wordvec = dict()
        print("Loading word vectors...")
        vectors = {}
        with open('%s' % path) as f:
            for line in f:
                s = line.strip()
                word = s[:s.find(' ')]
                vector = s[s.find(' ')+1:]
                vectors[word] = vector
        embed = []
        num_not_found, num_found = 0, 0
        for word in vocab_list:
            if word in vectors:
                vector = list(map(float, vectors[word].split()))
                num_found = num_found + 1
                vocab_wordvec[word] = None
            else:
                num_not_found = num_not_found + 1
                vector = np.random.random(self.FLAGS.dim_word) * 0.1
            embed.append(vector)
        print('%s words found in vocab' % num_found)
        print('%s words not found in vocab' % num_not_found)
        embed = np.array(embed, dtype=np.float32)
        return vocab_list, embed, vocab_wordvec

    def gen_batched_data(self, data):
        max_len_sen = min(max([len(item['sentence']) for item in data]), self.FLAGS.max_length_sen)
        
        def padding(sent, l, pad='_PAD'):
            return sent + [pad] * (l-len(sent))
        
        sentences, length_sen, aspects, labels = [], [], [], []
        for item in data:
            sentence = item['sentence']
            sentences.append(sentence[:max_len_sen] if len(sentence) > max_len_sen else padding(sentence, max_len_sen))
            length_sen.append(min(max_len_sen, len(sentence)))
            aspects.append(item['aspects'])
            labels.append(item['labels'])

        batched_data = {'batch_sentences': np.array(sentences), 'batch_length_sen': np.array(length_sen),
                        'batch_labels_asp': np.array(aspects),
                        'batch_labels': np.array(labels)
                        }
        return batched_data
