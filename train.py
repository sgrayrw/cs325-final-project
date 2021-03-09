import sys

from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.optimizers import SGD

from preprocess import *


def extract_feature(sentence, idxdict):
    preverb_sent, case_markers, final_verb, final_case_marker = sentence
    '''
    feature vector:
    [ unigrams | bigrams | case unigrams | case bigrams | final case ] -> {1, 0}
    '''
    ngrams = extract_ngrams((preverb_sent, case_markers, final_verb, final_case_marker))

    sections = ['unigrams', 'bigrams', 'case_unigrams', 'case_bigrams', 'final_case']

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
    train_y = np.array(train_y)
    dev_y = np.array(dev_y)
    try:
        history = model.fit(x_feature, train_y,
                            batch_size=Params.batch_size,
                            epochs=Params.epochs, verbose=0,
                            validation_data=(dev_x_feature, dev_y), )
        return history
    except ValueError as e:
        print(e)
    except:
        print('Unexpected error', sys.exc_info()[0])


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} version')
        exit(1)
    version = sys.argv[1]

    only_generate_data = False

    if only_generate_data:
        data = load_data(True)
        exit()

    data = load_data()
    x, y, idxdict = data['train_x'], data['train_y'], data['idxdict']

    split = int(len(x) / 9 * 8)
    train_x, train_y = x[:split], y[:split]
    dev_x, dev_y = x[split:], y[split:]

    train_x, train_y = shuffle_preverb_sent(train_x, train_y)

    feature_len = len(extract_feature(train_x[0], idxdict))

    train_from = 0
    if len(sys.argv) == 3 and sys.argv[2] == 'cont':
        model = load_model(f'models/model_{version}')
        train_from = load_obj(f'train_upto_{version}')
    else:
        model = Sequential()
        model.add(Dense(1, activation='sigmoid', input_shape=[feature_len]))
        model.compile(loss='binary_crossentropy',
                      optimizer=SGD(lr=Params.lr),
                      metrics=['accuracy'])

    print(f'start training from {train_from}')
    histories = []
    for i in tqdm(range(train_from, len(train_x), Params.train_batch_size)):
        history = train_model(train_x[i: i + Params.train_batch_size],
                              train_y[i: i + Params.train_batch_size],
                              dev_x[:Params.dev_size],
                              dev_y[:Params.dev_size],
                              idxdict, model)
        histories.append(history)

        # save the model every 128 iterations and record the index of the last successful batch,
        # in case the process is aborted
        if ((i - train_from) / Params.train_batch_size) % 128 == 0:
            model.save(f'models/model_{version}')
            print('training up to: ', i + Params.batch_size)
            save_obj(i + Params.train_batch_size, f'train_upto_{version}')

    save_obj(histories, f'histories_{version}')
