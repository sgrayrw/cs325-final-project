class Params:
    random_verb_pool = 500
    batch_size = 128
    epochs = 100
    lr = 0.05
    train_batch_size = 256  # how many examples to feed to model in one iter, this is different from `batch_size`
    dev_size = 1000
    feature_unigram_limit = 5000
    feature_bigram_limit = 10000
