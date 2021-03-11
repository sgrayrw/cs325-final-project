import random

import pandas as pd
from keras.preprocessing import sequence
from tqdm import tqdm

from util import *


def parse_file(filename, ignore_nonparticle=True):
    punc = '補助記号'
    verb = '動詞'
    noun = '名詞'
    adj = '形容詞'
    particle = '助詞'

    sentences = []

    with open(filename) as f:
        sent_count = nonparticle_count = 0

        for line in tqdm(f, 'parsing file'):
            preverb_sent, case_markers, final_verb = [], [], []
            final_case_marker = None
            verb_complete = verb_just_complete = False

            # split preverb sentence and final verb
            for triplet in reversed(line.split()):
                if triplet.count('/') != 2:
                    continue
                word, pos, pron = triplet.split('/')

                # discard punctuations
                if pos == punc:
                    continue
                # sentence likely not ending with verb
                if not verb_complete and pos in [noun, adj]:
                    break
                # 〜て〜
                if verb_just_complete and word in ['て', 'で'] and final_verb in [
                    'いる', 'いた', 'いない', 'いなかった', 'いく', 'いった',
                    'いかな', 'くる', 'きた', 'こな', 'しま'
                ]:
                    break
                # 漢語動詞
                if verb_just_complete and pos == noun and \
                        any(''.join(final_verb).startswith(prefix) for prefix in
                            ['する', 'した', 'しない', 'しなかった', 'して', 'され', 'させ', 'でき']):
                    break

                if not verb_complete:
                    final_verb.append(word)
                else:
                    preverb_sent.append(word)
                    if pos == particle:
                        case_markers.append(word)
                    if verb_just_complete:
                        verb_just_complete = False
                        final_case_marker = (word, pos)

                if not verb_complete and pos == verb:
                    verb_complete = verb_just_complete = True

            # exclude である
            if final_verb in ['ある', 'あった'] and final_case_marker[0] == 'で':
                continue

            if preverb_sent:
                # exclude sentences where final verb is not following a particle (case marker)
                if ignore_nonparticle and final_case_marker[1] != particle:
                    nonparticle_count += 1
                    continue
                sent_count += 1
                preverb_sent.reverse()
                case_markers.reverse()
                final_verb.reverse()
                sentences.append((preverb_sent, case_markers, final_verb, final_case_marker))

        nonparticle_ratio = nonparticle_count / sent_count if ignore_nonparticle else None
        return sentences, nonparticle_ratio


def prepare_negative_examples(dataset, all_verbs):
    i = 0
    pool = []
    for v in all_verbs:
        pool.append(v)
        i += 1
        if i >= Params.random_verb_pool:
            break

    x, y = [], []
    for preverb_sent, case_markers, final_verb, final_case_marker in dataset:
        x.append((preverb_sent, case_markers, final_verb, final_case_marker))
        y.append(1)

        negative_verb = final_verb
        while negative_verb[-1] == final_verb[-1]:
            negative_verb = random.choice(pool)
        x.append((preverb_sent, case_markers, negative_verb, final_case_marker))
        y.append(0)

    return x, np.array(y)


def prepare_multiple_choice(dataset, all_verbs):
    verb_idx = {}
    idx_verb = {}
    for i, (v, f) in enumerate(all_verbs.items()):
        verb_idx[v] = (i, f)
        idx_verb[i] = (v, f)

    def verb_choices(sent):
        verb = tuple(sent[2])
        idx, f = verb_idx[verb]
        l, r = idx - 1, idx + 1
        choices = [(verb, f)]
        while len(choices) < 6:  # two backup choices in case needed in the future
            if l >= 0:
                lv, lf = idx_verb[l]
            if r < len(idx_verb):
                rv, rf = idx_verb[r]
            if l < 0:
                choices.append((rv, rf))
                r += 1
            elif r >= len(idx_verb):
                choices.append((lv, lf))
                l -= 1
            elif abs(lf - f) > abs(rf - f):
                choices.append((rv, rf))
                r += 1
            else:
                choices.append((lv, lf))
                l -= 1
            if not qualified(choices):
                choices.pop()
        return choices

    # exclude verbs with the same first token (usually the verb stem)
    def qualified(choices):
        last_v = choices[-1]
        for v in choices[:-1]:
            if v[0][0] == last_v[0][0]:
                return False
        return True

    questions = []
    for example in tqdm(dataset, 'preparing multiple choice'):
        questions.append((example, verb_choices(example)))
    return questions


def index_all_verbs(sentences):
    all_verbs = defaultdict(int)
    for _, _, final_verb, _ in sentences:
        all_verbs[tuple(final_verb)] += 1
    all_verbs = sort_dict(all_verbs)
    return all_verbs


def extract_ngrams(sentence):
    preverb_sent, case_markers, final_verb, final_case_marker = sentence
    unigrams, bigrams, case_unigrams, case_bigrams, final_case = [], [], [], [], []

    '''
        {谷崎 潤一郎 は 数寄屋 を} (C) x {好 ん だ} (A):
        谷崎_好, 潤一郎_好 ... -> 1
        谷崎_X, 潤一郎_Y ... -> 0
    '''

    # C x A
    sent_bigrams, verb_bigrams = [], []
    prev_u1 = None
    for i, u1 in enumerate(preverb_sent):
        if prev_u1:
            sent_bigrams.append(f'{prev_u1}+{u1}')
        prev_u2 = None
        for u2 in final_verb:
            if i == 0 and prev_u2:
                verb_bigrams.append(f'{prev_u2}+{u2}')
            unigrams.append(f'{u1}_{u2}')
            prev_u2 = u2
        prev_u1 = u1
    for b1 in sent_bigrams:
        for b2 in verb_bigrams:
            bigrams.append(f'{b1}_{b2}')

    # case markers
    last_c = None
    for c in case_markers:
        case_unigrams.append(c)
        if last_c:
            case_bigrams.append((last_c, c))
        last_c = c

    # final case marker
    final_case.append(final_case_marker[0])

    return {
        'unigrams': unigrams,
        'bigrams': bigrams,
        'case_unigrams': case_unigrams,
        'case_bigrams': case_bigrams,
        'final_case': final_case
    }


def count_ngrams(sentences):
    freqdict = {d: defaultdict(int) for d in ['unigrams', 'bigrams', 'case_unigrams', 'case_bigrams', 'final_case']}

    '''
    {谷崎 潤一郎 は 数寄屋 を} (C) x {好 ん だ} (A):
        谷崎_好, 潤一郎_好 ... -> 1
        谷崎_X, 潤一郎_Y ... -> 0
    '''
    for preverb_sent, case_markers, final_verb, final_case_marker in tqdm(sentences, 'counting ngrams'):
        ngrams = extract_ngrams((preverb_sent, case_markers, final_verb, final_case_marker))
        for ngram, l in ngrams.items():
            for item in l:
                freqdict[ngram][item] += 1

    for k in freqdict:
        freqdict[k] = sort_dict(freqdict[k])
    return freqdict


# return a new set of sentences:
'''
original: 
S1 -> V1t
S1 -> V1f

new set:
S1 -> V1t
S1_shuffle -> V1t
S1 -> V1f
S1_shuffle -> V1f   (same shuffling order)
'''
def shuffle_preverb_sent(xs, ys):
    new_xs, new_ys = [], []
    for i, sent in tqdm(enumerate(xs), desc='shuffling preverb sent'):
        new_xs.append(sent)
        new_ys.append(ys[i])
        preverb_sent, case_markers, final_verb, final_case_marker = sent
        preverb_sent_shuffle = preverb_sent.copy()
        if i % 2 == 0:
            random.shuffle(preverb_sent_shuffle)
        else:
            preverb_sent_shuffle = new_xs[-2][0]  # look 2 ahead to follow the same shuffling order
        new_xs.append((preverb_sent_shuffle, case_markers, final_verb, final_case_marker))
        new_ys.append(ys[i])
    return new_xs, new_ys


def generate_data(reload_data_file=False):
    print('generating data ...')

    # parse from data file
    if reload_data_file:
        sentences, nonparticle_ratio = parse_file('kyoto-train.ja.pos')
        random.shuffle(sentences)
        save_obj(sentences, 'sentences')
    else:
        sentences = load_obj('sentences')

    # train, dev, test - 8:1:1
    all_verbs = index_all_verbs(sentences)
    # TODO: where should we sample negative verbs?
    length = len(sentences)
    train = sentences[:int(length * 0.8)]
    dev = sentences[int(length * 0.8): int(length * 0.9)]
    test = sentences[int(length * 0.9):]
    test.sort(key=lambda s: len(s[0]))  # evaluate.py assumes test is sorted by length asc

    # prepare negative verbs and multiple choice
    train_x, train_y = prepare_negative_examples(train, all_verbs)
    dev_x, dev_y = prepare_negative_examples(dev, all_verbs)
    test_x, test_y = prepare_negative_examples(test, all_verbs)
    questions = prepare_multiple_choice(test, all_verbs)

    # generate indexes for one-hot encoding
    freqdict = count_ngrams(train_x)
    idxdict = freqdict_to_inxdict(freqdict)

    # save data
    save_obj(train_x, 'logreg/train_x')
    save_obj(train_y, 'logreg/train_y')
    save_obj(dev_x, 'logreg/dev_x')
    save_obj(dev_y, 'logreg/dev_y')
    save_obj(test_x, 'logreg/test_x')
    save_obj(test_y, 'logreg/test_y')
    save_obj(questions, 'logreg/questions')
    save_obj(idxdict, 'logreg/idxdict')


def generate_lstm_data():
    sentences = load_obj('sentences')

    # index all characters
    all_chars = defaultdict(int)
    all_length = []
    for preverb, _, final_verb, _ in tqdm(sentences, 'Indexing all characters'):
        preverb_str = ''.join(preverb)
        final_verb_str = ''.join(final_verb)
        sent_str = preverb_str + final_verb_str
        all_length.append(len(sent_str))
        for char in sent_str:
            all_chars[char] += 1
    all_chars = sort_dict(all_chars)

    char_idx = {}
    for i, char in enumerate(all_chars):
        char_idx[char] = i

    # describe sent length - so we have an idea what the maxlen of a sequence should be
    print('Stat on sentence length:')
    print(pd.Series(all_length).describe())

    # convert data to sequences
    print('converting data into sequences...')
    train_x = load_obj('logreg/train_x')
    train_y = load_obj('logreg/train_y')
    dev_x = load_obj('logreg/dev_x')
    dev_y = load_obj('logreg/dev_y')
    test_x = load_obj('logreg/test_x')
    test_y = load_obj('logreg/test_y')
    questions = load_obj('logreg/questions')

    train_x = convert_sequence_dataset(train_x, char_idx)
    dev_x = convert_sequence_dataset(dev_x, char_idx)
    test_x = convert_sequence_dataset(test_x, char_idx)
    questions = convert_sequence_questions(questions, char_idx)

    # pad sequences
    print('padding sequences...')
    train_x = sequence.pad_sequences(train_x, maxlen=Params.lstm_maxlen)
    dev_x = sequence.pad_sequences(dev_x, maxlen=Params.lstm_maxlen)
    test_x = sequence.pad_sequences(test_x, maxlen=Params.lstm_maxlen)
    for i in range(len(questions)):
        questions[i] = sequence.pad_sequences(questions[i], maxlen=Params.lstm_maxlen)

    # save data
    print('saving data...')
    save_obj(train_x, 'lstm/train_x')
    save_obj(train_y, 'lstm/train_y')
    save_obj(dev_x, 'lstm/dev_x')
    save_obj(dev_y, 'lstm/dev_y')
    save_obj(test_x, 'lstm/test_x')
    save_obj(test_y, 'lstm/test_y')
    save_obj(questions, 'lstm/questions')
    save_obj(char_idx, 'lstm/char_idx')
    save_obj(len(char_idx), 'lstm/num_features')


# turn sentence into a sequence
def convert_sequence(sentence, char_idx):
    preverb, _, final_verb, _ = sentence
    preverb_str = ''.join(preverb)
    final_verb_str = ''.join(final_verb)
    sent_str = preverb_str + final_verb_str
    char_seq = []
    for char in sent_str:
        char_seq.append(char_idx[char])
    return char_seq


def convert_sequence_dataset(dataset, char_idx):
    result = []
    for sent in tqdm(dataset):
        result.append(convert_sequence(sent, char_idx))
    return np.array(result)


def convert_sequence_questions(questions, char_idx):
    result = []
    for sent, verb_choices in questions:
        candidate_sents = []
        for verb in verb_choices:
            sent = (sent[0], sent[1], verb[0], sent[3])
            candidate_sents.append(convert_sequence(sent, char_idx))
        result.append(np.array(candidate_sents))
    return result


if __name__ == '__main__':
    # generate_data(False)
    generate_lstm_data()
