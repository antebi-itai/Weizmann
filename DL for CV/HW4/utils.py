import pickle
import numpy as np
import torch
import os

SPECIAL_WORDS = {'PADDING': '<PAD>'}


def load_preprocess_data():
    return pickle.load(open('preprocess.p', mode='rb'))


def replace_tokens(sentence, token_dict):
    for key, token in token_dict.items():
        sentence = sentence.replace(' ' + token.lower(), key)

    sentence = sentence.replace('\n ', '\n')
    sentence = sentence.replace('( ', '(')
    return sentence


def batch_data(words, sequence_length, batch_size):
    x, y = [], []
    for index in range(0, (len(words) - sequence_length)):
        x.append(words[index: index + sequence_length])
        y.append(words[index + sequence_length])

    data = torch.utils.data.TensorDataset(torch.from_numpy(np.asarray(x)).long(), torch.from_numpy(np.asarray(y)).long())
    data_loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=batch_size)

    return data_loader


def save_model(filename, decoder):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(decoder, save_filename)


def load_model(filename):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    return torch.load(save_filename)


class Metric:
  def __init__(self):
    self.lst = 0.
    self.sum = 0.
    self.cnt = 0
    self.avg = 0.

  def update(self, val, cnt=1):
    self.lst = val
    self.sum += val * cnt
    self.cnt += cnt
    self.avg = self.sum / self.cnt

