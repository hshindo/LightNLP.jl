from gensim.models import word2vec
import logging
import sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

filename = sys.argv[1]
print("Processing " + filename)
data = word2vec.LineSentence(filename)
model = word2vec.Word2Vec(data, size=100, workers=4)
model.wv.save_word2vec_format('word2vec.model', binary=False)
