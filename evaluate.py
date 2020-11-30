from ngram_model import Ngram_model
# We want to try with words and POS_TAGS
# We want to try ngram sizes of 2-5 for each one
# We want to learn the best beta for each one
# Then report the perplexity

# So we can have a tuple of (0 : words, 1 : POS_TAG , n : ngram_size)

def open_corpus(path) :
    with open(path, 'r') as file :
        corpus = [line.rstrip('\n') for line in file]
    return corpus

train = open_corpus("train.txt")
tune = open_corpus("tune.txt")
test = open_corpus("test.txt")

for n in range(2, 6) :
    model = Ngram_model(train, n, 1)
    model.tune(tune)
    perplexity = model.perplexity(test)
    print("NGRAM SIZE: ", n)
    print("RESULT: ", perplexity)
    print()

# modes = {0 : "WORDS", 1 : "TAGS"}
# for mode in modes :
#     for n in range(3, 6) :
#         model = Ngram_model(train, n, mode)
#         model.tune(tune)
#         perplexity = model.perplexity(test)
#         print("MODE: ", modes[mode])
#         print("NGRAM SIZE: ", n)
#         print("RESULT: ", perplexity)
#         print()