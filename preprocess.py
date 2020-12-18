import random
import numpy as np
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


def prepare_negative_examples(dataset):
    all_verbs = defaultdict(int)
    for _, _, final_verb, _ in dataset:
        all_verbs[tuple(final_verb)] += 1
    all_verbs = sort_dict(all_verbs)

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


def prepare_multiple_choice(dataset):
    sentences = load_obj('sentences')
    all_verbs = defaultdict(int)
    for _, _, final_verb, _ in sentences:
        all_verbs[tuple(final_verb)] += 1
    all_verbs = sort_dict(all_verbs)

    verb_idx = {}
    idx_verb = {}
    for i, (v, f) in enumerate(all_verbs.items()):
        verb_idx[v] = (i, f)
        idx_verb[i] = (v, f)
    save_obj(verb_idx, 'verb_idx')
    save_obj(idx_verb, 'idx_verb')

    verb_idx = load_obj('verb_idx')
    idx_verb = load_obj('idx_verb')

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
    for example in tqdm(dataset):
        questions.append((example, verb_choices(example)))
    return questions


def extract_ngrams(sentence):
    preverb_sent, case_markers, final_verb, _ = sentence
    unigrams, bigrams, case_unigrams, case_bigrams = [], [], [], []

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

    return {
        'unigrams': unigrams,
        'bigrams': bigrams,
        'case_unigrams': case_unigrams,
        'case_bigrams': case_bigrams
    }


def count_ngrams(sentences):
    freqdict = {d: defaultdict(int) for d in ['unigrams', 'bigrams', 'case_unigrams', 'case_bigrams']}

    '''
    {谷崎 潤一郎 は 数寄屋 を} (C) x {好 ん だ} (A):
        谷崎_好, 潤一郎_好 ... -> 1
        谷崎_X, 潤一郎_Y ... -> 0
    '''
    for preverb_sent, case_markers, final_verb, _ in tqdm(sentences, 'counting ngrams'):
        ngrams = extract_ngrams((preverb_sent, case_markers, final_verb, _))
        for ngram, l in ngrams.items():
            for item in l:
                freqdict[ngram][item] += 1

    for k in freqdict:
        freqdict[k] = sort_dict(freqdict[k])
    return freqdict


def shuffle_preverb_sent(dataset):
    for i, sent in tqdm(enumerate(dataset), desc='shuffling preverb sent'):
        preverb_sent, case_markers, final_verb, final_case_marker = sent
        if i % 2 == 0:
            random.shuffle(preverb_sent)
        else:
            preverb_sent = dataset[i - 1][0]
        dataset[i] = (preverb_sent, case_markers, final_verb, final_case_marker)


def load_data(regenerate=True):
    print('loading data ...')
    if regenerate:
        sentences, nonparticle_ratio = parse_file('kyoto-train.ja.pos')
        random.shuffle(sentences)
        train = sentences[:int(len(sentences) * 0.9)]
        train_x, train_y = prepare_negative_examples(train)

        test = sentences[int(len(sentences) * 0.9):]
        test.sort(key=lambda s: len(s[0]))
        test_x, test_y = prepare_negative_examples(test)
        questions = prepare_multiple_choice(test)

        freqdict = count_ngrams(train_x)
        idxdict = freqdict_to_inxdict(freqdict)

        data = {'train_x': train_x, 'train_y': train_y,
                'test_x': test_x, 'test_y': test_y, 'questions': questions,
                'freqdict': freqdict, 'idxdict': idxdict}
        save_obj(data, 'data')
    else:
        data = load_obj('data')

    return data
