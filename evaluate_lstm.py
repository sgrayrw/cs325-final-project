from keras.models import load_model

from preprocess import *


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


if __name__ == '__main__':
    version = 'lstm'
    eval_mc_revealed(version)
