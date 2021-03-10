from keras.layers import Embedding, LSTM, Dropout, Dense
from keras.models import Sequential

from util import *


def create_model(num_features) -> Sequential:
    print('Creating model...')

    model = Sequential()
    model.add(Embedding(input_dim=num_features, output_dim=128, input_length=60))
    model.add(LSTM(units=128, activation='sigmoid', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(units=128, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def train_model(model: Sequential, train_x, train_y, dev_x, dev_y):
    print('training model...')

    return model.fit(train_x, train_y,
                     batch_size=64,
                     epochs=10,
                     validation_data=(dev_x, dev_y),
                     verbose=1)


if __name__ == '__main__':
    version = 'lstm'

    train_x = load_obj('lstm/train_x')
    train_y = load_obj('lstm/train_y')
    dev_x = load_obj('lstm/dev_x')
    dev_y = load_obj('lstm/dev_y')
    num_features = load_obj('lstm/num_features')

    model = create_model(num_features)
    hist = train_model(model, train_x, train_y, dev_x, dev_y)
    save_obj(hist, f'history/{version}')
    model.save(f'models/model_{version}')
