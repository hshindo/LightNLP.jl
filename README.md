# LightNLP
`LightNLP` is a natural language processing (NLP) toolkit for [Julia](http://julialang.org/).

[![Build Status](https://travis-ci.org/hshindo/LightNLP.jl.svg?branch=master)](https://travis-ci.org/hshindo/LightNLP.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/5wleyc2a1v1rldq8?svg=true)](https://ci.appveyor.com/project/hshindo/lightnlp-jl)

## Requirements
- Julia 0.6.x

## Installation
```julia
julia> Pkg.clone("https://github.com/hshindo/LightNLP.jl.git")
```

## Named Entity Recognition (NER)
### Training
First, download [pre-trained word embeddings](https://cl.naist.jp/~shindo/glove.6B.100d.h5) and put it in `.data/`.  
Then, run the script:
```
julia train.jl <trainfile> <testfile>
```

### Decoding
```
julia test.jl <modelfile> <testfile>
```
