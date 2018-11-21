using HDF5

#file = ".data/PTBLM/ptb.test.txt"
#x = read(file)
#h5write("ptblm.h5", "test", x)
file = ".data/PTBLM/ptb.train.txt"
#x = read(file)
#h5write("ptblm.h5", "train", x)
strs = String[]
for line in open(readlines,file)
    elems = collect(split(chomp(line)))
    append!(strs, elems)
end
h5write("x.h5", "s", strs)
