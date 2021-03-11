from keras.models import load_model

from preprocess import *


def eval_binary_revealed(version, num_buckets=5):
    model = load_model(f'models/model_{version}')

    # see comment in eval_mc_revealed for reason to use logreg test data
    test_x = load_obj('logreg/test_x')
    test_y = load_obj('logreg/test_y')

    buckets = [[] for _ in range(num_buckets)]
    y = []

    for i, sent in tqdm(enumerate(test_x), 'assigning buckets'):
        preverb = sent[0]
        if len(preverb) < num_buckets:
            continue

        y.append(test_y[i])
        chop_len = len(preverb) // num_buckets
        for bucket in range(num_buckets):
            preverb_new = preverb[:chop_len * (1 + bucket)]
            sent_new = (preverb_new, sent[1], sent[2], sent[3])
            buckets[bucket].append(sent_new)

    test_y = np.array(y)
    char_idx = load_obj('lstm/char_idx')
    percent_revealed, counts, accs = [], [], []
    for bucket in range(num_buckets):
        test_x = convert_sequence_dataset(buckets[bucket], char_idx)
        test_x = sequence.pad_sequences(test_x, maxlen=Params.lstm_maxlen)
        acc = model.evaluate(test_x, test_y, verbose=1)[1]
        accs.append(acc)
        percent_revealed.append((bucket + 1) * 1 / num_buckets)
        counts.append(len(buckets[bucket]))

    binary_result = {'percent_revealed': {'val': percent_revealed, 'accs': accs, 'counts': counts}}
    save_obj(binary_result, f'evaluated_result/{version}/binary_percent_result')


def eval_mc_revealed(version, num_choices=4, num_buckets=5):
    model = load_model(f'models/model_{version}')

    # using logreg/questions instead of lstm/questions b/c in order to
    # calculate percent revealed, we need to know about the original preverb|verb boundary
    # whereas lstm/questions only contain padded sequence data
    questions = load_obj('logreg/questions')

    buckets = [[] for _ in range(num_buckets)]

    for sent, verb_choices in tqdm(questions, 'assigning buckets'):
        preverb = sent[0]
        if len(preverb) < num_buckets:
            continue

        chop_len = len(preverb) // num_buckets
        for bucket in range(num_buckets):
            preverb_new = preverb[:chop_len * (1 + bucket)]
            sent_new = (preverb_new, sent[1], sent[2], sent[3])
            buckets[bucket].append((sent_new, verb_choices))

    char_idx = load_obj('lstm/char_idx')
    percent_revealed, counts, accs = [], [], []
    for bucket in range(num_buckets):
        acc = predict_questions(f'bucket {bucket + 1}/{num_buckets}', model, buckets[bucket], num_choices, char_idx)
        accs.append(acc)
        percent_revealed.append((bucket + 1) * 1 / num_buckets)
        counts.append(len(buckets[bucket]))

    mc_result = {'percent_revealed': {'val': percent_revealed, 'accs': accs, 'counts': counts}}
    save_obj(mc_result, f'evaluated_result/{version}/mc_percent_result')


def predict_questions(prompt, model, questions, num_choices, char_idx):
    questions = convert_sequence_questions(questions, char_idx)
    for i in range(len(questions)):
        questions[i] = sequence.pad_sequences(questions[i], maxlen=Params.lstm_maxlen)

    correct = wrong = 0
    for question in tqdm(questions, prompt):
        question = question[:num_choices]
        prediction = model.predict(question)
        if np.argmax(prediction) == 0:
            correct += 1
        else:
            wrong += 1
    assert (correct + wrong == len(questions))
    return correct / (correct + wrong)


if __name__ == '__main__':
    version = 'lstm'
    eval_binary_revealed(version)
    # eval_mc_revealed(version)
