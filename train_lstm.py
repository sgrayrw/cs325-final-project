from keras.layers import Embedding, LSTM, Dropout, Dense
from keras.models import Sequential

from Params import Params


def create_model(num_features):
    print('Creating model...')
    model = Sequential()
    model.add(Embedding(num_features, Params.lstm_breadth, Params.lstm_maxlen))
    model.add(LSTM(Params.lstm_breadth,
                   activation='sigmoid',
                   inner_activation='hard_sigmoid',
                   return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(Params.lstm_breadth,
                   activation='sigmoid',
                   inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


if __name__ == '__main__':
    version = 'lstm'

