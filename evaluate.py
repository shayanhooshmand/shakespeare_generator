from ngram_model import Ngram_model
def open_corpus(path) :
    with open(path, 'r') as file :
        corpus = [line.rstrip('\n') for line in file]
    return corpus

train = open_corpus("train.txt")
tune = open_corpus("tune.txt")
test = open_corpus("test.txt")

modes = {0 : "WORDS", 1 : "TAGS"}
for mode in modes :
    for n in range(2, 6) :
        model = Ngram_model(train, n, mode)
        # model.tune(tune)
        perplexity = model.perplexity(test)
        print("MODE: ", modes[mode])
        print("NGRAM SIZE: ", n)
        print("RESULT: ", perplexity)
        print()