from keras.models import load_model
from sklearn.metrics import confusion_matrix
from preprocess import *
from train import extract_feature


def case_density(sent):
    return len(sent[1]) / len(sent[0])


def verb_freq(question):
    return question[1][0][1]


def eval_binary_classification(ver):
    data = load_obj('data')
    model = load_model(f'model{ver}')
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
    binary_result['len'] = {'lens': lens, 'accs': accs, 'counts': counts}
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
    binary_result['case_den'] = {'dens': dens, 'accs': accs, 'counts': counts}

    save_obj(binary_result, f'binary_result{ver}')


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


def eval_multiple_choice(ver, num_choice=4):
    data = load_obj('data')
    questions = data['questions']
    model = load_model(f'model{ver}')

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
    mc_result['len'] = {'lens': lens, 'accs': accs, 'counts': counts}

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
    mc_result['case_den'] = {'dens': dens, 'accs': accs, 'counts': counts}

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
    mc_result['freq'] = {'freqs': freqs, 'accs': accs, 'counts': counts}

    save_obj(mc_result, f'mc_result{ver}')


if __name__ == '__main__':
    ver = ''
    eval_binary_classification(ver)
    eval_multiple_choice(ver)
