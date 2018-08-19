const CONFIG = Dict()
CONFIG["wordvec_file"] = ".data/glove.6B.100d.h5"
CONFIG["train_file"] = ".data/eng.train.BIOES"
CONFIG["dev_file"] = ".data/eng.testa.BIOES"
CONFIG["test_file"] = ".data/eng.testb.BIOES"
CONFIG["nepochs"] = 50
CONFIG["learning_rate"] = 0.005
#CONFIG["learning_rate"] = 0.0015
CONFIG["batchsize"] = 10
CONFIG["training"] = true
