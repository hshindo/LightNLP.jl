# Named Entity Recognition
This is an example of Named Entity Recognition (NER) with neural networks.  

## Training
First, download [pre-trained word embeddings](https://cl.naist.jp/~shindo/glove.6B.100d.h5) and put it in `.data/`.  
Then, run the script:
```
julia train.jl <trainfile> <testfile>
```

## Decoding
```
julia test.jl <modelfile> <testfile>
```
