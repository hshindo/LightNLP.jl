function train(dec::Decoder, trainfile::String, testfile::String)
    traindata = readdata!(dec, trainfile)
    testdata = readdata!(dec, testfile)

    info("#Training examples:\t$(length(traindata))")
    info("#Testing examples:\t$(length(testdata))")
    info("#Words:\t$(length(dec.worddict))")
    info("#Chars:\t$(length(dec.chardict))")
    info("#Tags:\t$(length(dec.tagset.tag2id))")
    testdata = batch(testdata, 100)

    batchsize = 10
    opt = SGD()
    for epoch = 1:50
        println("epoch:\t$epoch")
        opt.rate = 0.0005 * batchsize / sqrt(batchsize) / (1 + 0.05*(epoch-1))
        println("learning rate: $(opt.rate)")

        shuffle!(traindata)
        batches = batch(traindata, batchsize)
        prog = Progress(length(batches))
        loss = 0.0
        for i in 1:length(batches)
            w, c, t = batches[i]
            h = dec.model(w, c)
            y = softmax_crossentropy(t, h)
            loss += sum(y.data)
            params = gradient!(y)
            foreach(opt, params)
            ProgressMeter.next!(prog)
        end
        loss /= length(batches)
        println("Loss:\t$loss")

        # test
        println("Testing...")
        pred = Int[]
        gold = Int[]
        for (w,c,t) in testdata
            y = dec.model(w, c)
            append!(pred, vec(argmax(y.data,1)))
            append!(gold, t.data)
        end
        length(pred) == length(gold) || throw("Length mismatch: $(length(pred)), $(length(gold))")

        ranges_p = decode(dec.tagset, pred)
        ranges_g = decode(dec.tagset, gold)
        fscore(ranges_g, ranges_p)
        println()
    end
end

function fscore{T}(golds::Vector{T}, preds::Vector{T})
    set = intersect(Set(golds), Set(preds))
    count = length(set)
    prec = round(count/length(preds), 5)
    recall = round(count/length(golds), 5)
    fval = round(2*recall*prec/(recall+prec), 5)
    println("Prec:\t$prec")
    println("Recall:\t$recall")
    println("Fscore:\t$fval")
end
