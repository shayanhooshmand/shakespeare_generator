import nltk
from collections import defaultdict, Counter

def get_lexicon(corpus, minimum) :
    """
    returns the set of words in a corpus with count greater
    than 1
    """
    counts = Counter(corpus)
    return set(word for word in counts if counts[word] > minimum)

def get_ngrams(line, n) :
    """
    returns list of ngrams given a corpus and size n
    """
    bookended = [('START', 'START')] * max(1, n-1) + line + [('STOP', 'STOP')]
    return [tuple(bookended[i:i+n]) for i in range(len(bookended) - (n - 1))]

class Ngram_model :

    def __init__(self, corpus, n=3, POS_TAG=1) :
        """
        corpus : list of strings, each string a Shakespeare line
        n : size of ngram for model
        POS_TAG : 1 means use POS_TAGS for ngrams, 0 means use words
        """
        self.n = n
        self.POS_TAG = POS_TAG

        corpus = [nltk.word_tokenize(line) for line in corpus]

        self.lexicon = get_lexicon([word for line in corpus for word in line], 0)
        self.lexicon.add('START')
        self.lexicon.add('STOP')
        self.lexicon.add('UNK')

        self.corpus = [nltk.pos_tag(line) for line in corpus]
        #replace words that have small counts with 'UNK' token
        for line in self.corpus :
            for i, (word, tag) in enumerate(line) :
                if word not in self.lexicon and not POS_TAG :
                    line[i] = ('UNK', 'UNK')
        
        for line in self.corpus :
            print(line)
        
        #count up all the k-grams from unigram to ngram
        self.ngram_counts = [self.count_ngrams(i) for i in range(1, self.n+1)]
    
    def count_ngrams(self, n) :

        ngram_counts = defaultdict(int)
        START = tuple(['START'] * n)
        for line in self.corpus :
            for tuple_ngram in get_ngrams(line, n) :
                #pick either word or tag to count depending on instance of model
                ngram = tuple([pair[self.POS_TAG] for pair in tuple_ngram])
                ngram_counts[ngram] += 1
                #add ngram full of starts for each beginning of sentence
                #for when we calculate likelihoods later
                if n > 1 :
                    if ngram[0] == 'START' and ngram[1] != ngram[0] :
                        ngram_counts[START] += 1
        
        return ngram_counts


training_file = "./training/formatted_training/test_training.txt"
with open(training_file, 'r') as file :
    corpus = [line.rstrip('\n') for line in file]

model = Ngram_model(corpus, 3, 0)
for dictionary in model.ngram_counts :
    print(dictionary)
    print()
