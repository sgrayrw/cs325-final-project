import numpy as np
from tqdm import tqdm
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD

from util import *
from Params import Params
from preprocess import *


def extract_feature(sentence, idxdict, use_final_case=True):
    preverb_sent, case_markers, final_verb, final_case_marker = sentence
    '''
    feature vector:
    [ unigrams | bigrams | case unigrams | case bigrams | final case ] -> {1, 0}
    '''
    ngrams = extract_ngrams((preverb_sent, case_markers, final_verb, final_case_marker))

    sections = ['unigrams', 'bigrams', 'case_unigrams', 'case_bigrams']
    if use_final_case:
        sections.append('final_case')

    features = []
    for section in sections:
        d = idxdict[section]
        v = np.zeros(len(d))
        for item in ngrams[section]:
            if item in d:
                v[d[item]] = 1
        features.append(v)
    return np.concatenate(features)


def train_model(train_x, train_y, dev_x, dev_y, idxdict, model):
    x_feature = []
    dev_x_feature = []
    for example in train_x:
        x_feature.append(extract_feature(example, idxdict))
    for example in dev_x:
        dev_x_feature.append(extract_feature(example, idxdict))
    x_feature = np.vstack(x_feature)
    dev_x_feature = np.vstack(dev_x_feature)
    return model.fit(x_feature,
                     train_y,
                     batch_size=Params.batch_size,
                     epochs=Params.epochs,
                     verbose=2,
                     validation_data=(dev_x_feature, dev_y)).history


if __name__ == '__main__':
    only_generate_data = False

    if only_generate_data:
        data = load_data(True)
        exit()

    data = load_data()
    x, y, idxdict = data['train_x'], data['train_y'], data['idxdict']

    split = int(len(x) / 9 * 8)
    train_x, train_y = x[:split], y[:split]
    dev_x, dev_y = x[split:], y[split:]

    feature_len = len(extract_feature(train_x[0], idxdict))

    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_shape=[feature_len]))
    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=Params.lr),
                  metrics=['accuracy'])

    history = []
    for i in tqdm(range(0, len(train_x), Params.train_batch_size)):
        history.append(train_model(train_x[i: i + Params.train_batch_size],
                                   train_y[i: i + Params.train_batch_size],
                                   dev_x[:Params.dev_size],
                                   dev_y[:Params.dev_size],
                                   idxdict,
                                   model))
        model.save('model')
    save_obj(history, 'history')
