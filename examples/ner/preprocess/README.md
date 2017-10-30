## 1. Tokenization and Normalization
Tokenization utilizes Stanford NER tokenizer.  
Normalization performs the following processes.
* Lowercase  
ex. Obama → obama
* Replace number with 0  
ex. 123.45 → 000.00
* Replace infrequent words with the special token: `UNKNOWN`.  
The default threshold is set to 5.

First, download [Stanford NER](https://nlp.stanford.edu/software/CRF-NER.shtml#Download)
and put `stanford-ner.jar` in this directory.  
Then, run the following command:
```
julia normalize_text.jl <input>
```
where `<input>` is an output from `xml2txt`.

## 2. Word2Vec
Learns word representation from texts by using gensim word2vec.  
The word representation is saved as HDF5 format.

First, install python3 and the following libraries:
```
pip install numpy cython gensim
```
Then, run the following command:
```
julia word2vec.jl <input>
```
where `<input>` is an output from `normalize_text.jl`
