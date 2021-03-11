import sys

from keras.models import load_model
from sklearn.metrics import confusion_matrix

from preprocess import *
from train_logreg import extract_feature


def case_density(sent):
    return len(sent[1]) / len(sent[0])


def verb_freq(question):
    return question[1][0][1]


def eval_binary_classification(version):
    data = load_obj('data')
    model = load_model(f'models/model_{version}')
    test_x, test_y = data['test_x'], data['test_y']

    binary_result = {}

    sent_len = 1
    i = 0
    lens, counts, accs = [], [], []
    probas = None
    while i < len(test_x):
        x_feature, y = [], []
        while i < len(test_x) and len(test_x[i][0]) <= sent_len:
            x_feature.append(extract_feature(test_x[i], data['idxdict']))
            y.append(test_y[i])
            i += 1
        if x_feature:
            x_feature = np.vstack(x_feature)
            y = np.array(y)
            score = model.evaluate(x_feature, y)
            p = model.predict(x_feature)
            if probas is None:
                probas = p
            else:
                probas = np.concatenate((probas, p))
            lens.append(sent_len)
            accs.append(score[1])
            counts.append(len(x_feature))
        sent_len += 1
    probas = np.squeeze(probas)
    y_pred = np.round(probas)
    conf = confusion_matrix(test_y, y_pred)
    binary_result['len'] = {'val': lens, 'accs': accs, 'counts': counts}
    binary_result['conf'] = conf

    test_x, test_y = zip(*sorted(zip(test_x, test_y), key=lambda pair: case_density(pair[0])))
    case_den = 0.02
    i = 0
    dens, counts, accs = [], [], []
    while i < len(test_x):
        x_feature, y = [], []
        while i < len(test_x) and case_density(test_x[i]) <= case_den:
            x_feature.append(extract_feature(test_x[i], data['idxdict']))
            y.append(test_y[i])
            i += 1
        if x_feature:
            x_feature = np.vstack(x_feature)
            y = np.array(y)
            score = model.evaluate(x_feature, y)
            dens.append(case_den)
            accs.append(score[1])
            counts.append(len(x_feature))
        case_den += 0.02
    binary_result['case_den'] = {'val': dens, 'accs': accs, 'counts': counts}

    save_obj(binary_result, f'evaluated_result/{version}/binary_result')


def predict_questions(model, qs, idxdict, num_choice):
    correct = wrong = 0
    for sent, choices in qs:
        sent = list(sent)
        features = []
        for choice in choices[:num_choice]:
            sent[2] = choice[0]
            features.append(extract_feature(sent, idxdict))
        features = np.vstack(features)
        prediction = model.predict(features)
        if np.argmax(prediction) == 0:
            correct += 1
        else:
            wrong += 1
    assert (correct + wrong == len(qs))
    return correct / (correct + wrong)


def eval_multiple_choice(version, num_choice=4):
    data = load_obj('data')
    questions = data['questions']
    model = load_model(f'models/model_{version}')

    mc_result = {}

    # sent len
    sent_len = 1
    i = 0
    lens, counts, accs = [], [], []
    pbar = tqdm(total=len(questions), desc='mc sent len')
    while i < len(questions):
        qs = []
        while i < len(questions) and len(questions[i][0][0]) <= sent_len:
            qs.append(questions[i])
            i += 1
            pbar.update()
        if qs:
            acc = predict_questions(model, qs, data['idxdict'], num_choice)
            lens.append(sent_len)
            accs.append(acc)
            counts.append(len(qs))
        sent_len += 1
    pbar.close()
    mc_result['len'] = {'val': lens, 'accs': accs, 'counts': counts}

    # case density
    questions.sort(key=lambda q: case_density(q[0]))
    case_den = 0.02
    i = 0
    dens, counts, accs = [], [], []
    pbar = tqdm(total=len(questions), desc='mc case den')
    while i < len(questions):
        qs = []
        while i < len(questions) and case_density(questions[i][0]) <= case_den:
            qs.append(questions[i])
            i += 1
            pbar.update()
        if qs:
            acc = predict_questions(model, qs, data['idxdict'], num_choice)
            dens.append(case_den)
            accs.append(acc)
            counts.append(len(qs))
        case_den += 0.02
    pbar.close()
    mc_result['case_den'] = {'val': dens, 'accs': accs, 'counts': counts}

    # verb freq
    questions.sort(key=lambda q: verb_freq(q))
    freq = 50
    i = 0
    freqs, counts, accs = [], [], []
    pbar = tqdm(total=len(questions), desc='mc verb freq')
    while i < len(questions):
        qs = []
        while i < len(questions) and verb_freq(questions[i]) <= freq:
            qs.append(questions[i])
            i += 1
            pbar.update()
        if qs:
            acc = predict_questions(model, qs, data['idxdict'], num_choice)
            freqs.append(freq)
            accs.append(acc)
            counts.append(len(qs))
        freq += 50
    pbar.close()
    mc_result['freq'] = {'val': freqs, 'accs': accs, 'counts': counts}

    save_obj(mc_result, f'evaluated_result/{version}/mc_result')


def eval_binary_revealed(version, num_buckets=5):
    data = load_obj('data')
    model = load_model(f'models/model_{version}')
    test_x, test_y = data['test_x'], data['test_y']

    binary_result = {}

    sent_len = 1
    i = 0
    percent_revealed, counts, accs = [], [], []
    pbar = tqdm(total=len(test_x), desc='binary percent revealed')
    while i < len(test_x):
        x_feature = [[] for _ in range(num_buckets)]
        y = [[] for _ in range(num_buckets)]

        empty = True
        while i < len(test_x) and len(test_x[i][0]) <= sent_len:
            if num_buckets <= len(test_x[i][0]):
                empty = False
                sent = test_x[i]
                chop_len = len(sent[0]) // num_buckets
                for bucket in range(num_buckets):
                    preverb = sent[0][:chop_len * (1 + bucket)]
                    feature = extract_feature((preverb, sent[1], sent[2], sent[3]), data['idxdict'])
                    x_feature[bucket].append(feature)
                    y[bucket].append(test_y[i])
            i += 1
            pbar.update()

        if not empty:
            for bucket in range(num_buckets):
                features = np.vstack(x_feature[bucket])
                labels = np.array(y[bucket])
                acc = model.evaluate(features, labels, verbose=0)[1]
                accs.append(acc)
                percent_revealed.append((bucket + 1) * 1 / num_buckets)
                counts.append(len(features))

        sent_len += 1

    binary_result['percent_revealed'] = {'val': percent_revealed, 'accs': accs, 'counts': counts}
    save_obj(binary_result, f'evaluated_result/{version}/binary_percent_result')


def eval_mc_revealed(version, num_choice=4, num_buckets=5):
    data = load_obj('data')
    questions = data['questions']
    model = load_model(f'models/model_{version}')

    mc_result = {}

    sent_len = 1
    i = 0
    percent_revealed, counts, accs = [], [], []
    pbar = tqdm(total=len(questions), desc='mc percent revealed')
    while i < len(questions):
        qs = [[] for _ in range(num_buckets)]
        empty = True
        while i < len(questions) and len(questions[i][0][0]) <= sent_len:
            if num_buckets <= len(questions[i][0][0]):
                empty = False
                sent = questions[i][0]
                chop_len = len(sent[0]) // num_buckets
                for bucket in range(num_buckets):
                    preverb = sent[0][:chop_len * (1 + bucket)]
                    sent_new = (preverb, sent[1], sent[2], sent[3])
                    qs[bucket].append((sent_new, questions[i][1]))
            i += 1
            pbar.update()

        if not empty:
            for bucket in range(num_buckets):
                acc = predict_questions(model, qs[bucket], data['idxdict'], num_choice)
                accs.append(acc)
                percent_revealed.append((bucket + 1) * 1 / num_buckets)
                counts.append(len(qs[bucket]))

        sent_len += 1

    mc_result['percent_revealed'] = {'val': percent_revealed, 'accs': accs, 'counts': counts}
    save_obj(mc_result, f'evaluated_result/{version}/mc_percent_result')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} version')
        exit(1)
    version = sys.argv[1]

    eval_binary_classification(version)
    eval_multiple_choice(version)
    # eval_binary_revealed(version)
    # eval_mc_revealed(version)
