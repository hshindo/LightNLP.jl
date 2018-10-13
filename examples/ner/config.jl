const config = (
    wordvec_file = ".data/glove.6B.100d.h5",
    train_file = ".data/eng.train.BIOES",
    dev_file = ".data/eng.testa.BIOES",
    test_file = ".data/eng.testb.BIOES",
    nepochs = 1,
    learning_rate = 0.0015,
    batchsize = 10,
    training = true,
    device = 0
)
