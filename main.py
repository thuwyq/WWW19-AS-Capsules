# coding: utf-8
#
# Copyright 2018 Yequan Wang
# Author: Yequan Wang (tshwangyequan@gmail.com)
#

import time
import random
import argparse
import torch
import numpy as np

from datamanager import DataManager
from bridgeModel import bridgeModel
from AspEvaluation import evaluateModel

parser = argparse.ArgumentParser()
parser.add_argument('--voc_size', type=int, default=32768)
parser.add_argument('--dim_word', type=int, default=300, choices=[300])
parser.add_argument('--dim_hidden', type=int, default=256, choices=[256])
parser.add_argument('--n_layer', type=int, default=1)
parser.add_argument('--n_label', type=int, default=3, choices=[3])
parser.add_argument('--n_aspect', type=int, default=5, choices=[5])
parser.add_argument('--bidirectional', type=bool, default=True)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--lr_word_vector', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--embed_dropout', type=float, default=0.5)
parser.add_argument('--cell_dropout', type=float, default=0.5)
parser.add_argument('--final_dropout', type=float, default=0.5)
parser.add_argument('--lambda1', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_length_sen', type=int, default=64)
parser.add_argument('--iter_num', type=int, default=8*320)
parser.add_argument('--per_checkpoint', type=int, default=8)
parser.add_argument('--seed', type=int, default=2018)
parser.add_argument('--rnn_type', type=str, default="LSTM", choices=["LSTM", "GRU"])
parser.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "Adadelta", "RMSprop", "Adagrad"])
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--path_wordvec', type=str, default='vectors.glove.840B.txt')
parser.add_argument('--name_model', type=str, default='AS-Capsules-master')
FLAGS = parser.parse_args()
print(FLAGS)

np.random.seed(FLAGS.seed)
random.seed(FLAGS.seed)
torch.manual_seed(FLAGS.seed)
# torch.backends.cudnn.enabled = False
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed(FLAGS.seed)


def train(model, datamanager, data_train):
    selected_data = [random.choice(data_train) for i in range(FLAGS.batch_size)]
    batched_data = datamanager.gen_batched_data(selected_data)
    loss, _, _, _, _ = model.stepTrain(batched_data)
    return loss

num_loss = 3
def evaluate(model, datamanager, data_):
    loss = np.zeros((num_loss, ))
    st, ed, times = 0, FLAGS.batch_size, 0
    y_pred_asp, y_true_asp = [], []
    y_pred_sent, y_true_sent = [], []
    while st < min(len(data_), 1e4):
        selected_data = data_[st:ed]
        batched_data = datamanager.gen_batched_data(selected_data)
        _loss, pred_asp, pred_sent, att_asp, att_sen = model.stepTrain(batched_data, inference=True)
        y_pred_asp.extend(pred_asp > 0.5)
        y_true_asp.extend(batched_data['batch_labels_asp'])
        y_pred_sent.extend(np.argmax(pred_sent, axis=-1))
        y_true_sent.extend(batched_data['batch_labels'])
        loss += _loss
        st, ed = ed, ed + FLAGS.batch_size
        times += 1
    y_pred_asp, y_true_asp = np.array(y_pred_asp), np.array(y_true_asp)
    y_pred_sent, y_true_sent = np.array(y_pred_sent), np.array(y_true_sent)
    dict_eva = evaluateModel(y_true_asp, y_pred_asp+.0, y_true_sent, y_pred_sent)
    return loss/times, dict_eva


class AS_Capsules(object):
    def __init__(self):
        self.dataset_name = ('train', 'valid', 'test')
        self.datamanager = DataManager(FLAGS)
        self.data = {}
        for tmp in self.dataset_name:
            self.data[tmp] = self.datamanager.load_data(FLAGS.data_dir, '%s.txt' % tmp)
        vocab, embed, vocab_dict = self.datamanager.build_vocab('%s/%s' % (FLAGS.data_dir, FLAGS.path_wordvec), 
                                                            self.data['train'] + self.data['valid'] + self.data['test'])

        print('model parameters: %s' % str(FLAGS))
        print("Use cuda: %s" % use_cuda)
        for name in self.dataset_name:
            print('Dataset Statictis: %s data: %s' % (name, len(self.data[name])))

        self.model = bridgeModel(
                FLAGS.dim_word,
                FLAGS.dim_hidden, 
                FLAGS.n_layer,
                FLAGS.n_label,
                FLAGS.n_aspect,
                batch_size=FLAGS.batch_size,
                max_length_sen=FLAGS.max_length_sen,
                learning_rate=FLAGS.learning_rate,
                lr_word_vector=FLAGS.lr_word_vector,
                weight_decay=FLAGS.weight_decay,
                vocab=vocab,
                embed=embed,
                embed_dropout_rate=FLAGS.embed_dropout,
                cell_dropout_rate=FLAGS.cell_dropout,
                final_dropout_rate=FLAGS.final_dropout,
                bidirectional=FLAGS.bidirectional,
                optim_type=FLAGS.optim_type,
                rnn_type=FLAGS.rnn_type,
                lambda1=FLAGS.lambda1,
                use_cuda=use_cuda)

    def train(self):
        loss_step, time_step = np.ones((num_loss, )), 0
        start_time = time.time()    
        for step in range(FLAGS.iter_num):
            if step % FLAGS.per_checkpoint == 0:
                show = lambda a: '[%s]' % (' '.join(['%.4f' % x for x in a]))
                time_step = time.time() - start_time
                print("------------------------------------------------------------------")
                print('Time of iter training %.2f s' % time_step)
                print("On iter step %s:, global step %d Loss-step %s" % (step/FLAGS.per_checkpoint, step, show(np.exp(loss_step))))
                # self.model.save_model("%s/%s" % ("./model", FLAGS.name_model), int(step/FLAGS.per_checkpoint))

                for name in self.dataset_name:
                    loss, dict_eva = evaluate(self.model, self.datamanager, self.data[name])
                    print('In dataset %s: Loss is %s, Accu-Asp is %s, F1-Asp is %s' % (name, show(np.exp(loss)), show(dict_eva['Asp']['acc']), show(dict_eva['Asp']['f1'])))
                    print('Loss is %s, Accu-Sen is %.4f, F1-Sen is %s' % (show(np.exp(loss)), dict_eva['Sen']['acc'], show(dict_eva['Sen']['f1'])))
                    print('Loss is %s, Accu-All is %.4f, F1-All is %s' % (show(np.exp(loss)), dict_eva['All']['acc'], show(dict_eva['All']['f1'])))
                    print('For Asp, Micro-F1 is %s' % dict_eva['micro_F1_Asp'])
                    print('For Sen, C_M is \n%s' % dict_eva['Sen']['c_m'])
                    print('For All, C_M is \n%s' % dict_eva['All']['c_m'])
                
                start_time = time.time()
                loss_step = np.zeros((num_loss, ))
                
            loss_step += train(self.model, self.datamanager, self.data['train']) / FLAGS.per_checkpoint


if __name__ == "__main__":
    as_c = AS_Capsules()
    as_c.train()
