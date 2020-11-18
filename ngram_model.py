import nltk
from collections import defaultdict, Counter

def get_lexicon(corpus, minimum) :
    """
    returns the set of words in a corpus with count greater
    than 1
    """
    counts = Counter(corpus)
    return set(word for word in counts if counts[word] > minimum)

def pre_process(corpus, lexicon) :
    """
    replace words in corpus with UNK token if word
    not in lexicon
    """
    for line in corpus :
        for i, (word, tag) in enumerate(line) :
            if word not in lexicon :
                line[i] = ('UNK', 'UNK')

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
        #only if counting words and not tags
        if not POS_TAG : pre_process(self.corpus, self.lexicon)
        
        #initialize total number of tokens
        self.M = 0
        #count up all the k-grams from unigram to ngram
        self.ngram_counts = [self.count_ngrams(i) for i in range(1, self.n+1)]

        return

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
                #if we are counting unigrams then also use this loop to calculate
                #count up total number of tokens
                elif n == 1 :
                    self.M += 1
        
        return ngram_counts
    
    def backoff_prob(self, ngram, beta) :
        """
        returns backoff probability of an ngram
        beta : discount with 0<beta<1
        """
        
        if not 0 < beta < 1 :
            print("ERR: Beta must be between 0 and 1 for backoff calculation.")
            exit(1)
        
        def get_alpha(ng) :
            mass = 0
            for n_plus_1_gram in self.ngram_counts[len(ng)] :
                if n_plus_1_gram[:-1] == ng :
                    raw_count = self.ngram_counts[len(ng)][n_plus_1_gram]
                    if raw_count > 0 :
                        mass += raw_count - beta
            return 1 - (mass / self.ngram_counts[len(ng)-1][ng])

        #base : unigram
        if len(ngram) == 1 :
            return self.ngram_counts[0][ngram] / self.M

        ngram_count = self.ngram_counts[len(ngram)-1][ngram]
        #base : count is greater than 0
        if ngram_count > 0 :
            return (ngram_count - beta) / self.ngram_counts[len(ngram)-2][ngram[:-1]]
        
        #otherwise backoff
        alpha = get_alpha(ngram[:-1])
        normalizer = 0
        #construct all the other unseen ngrams in this context
        for word in self.lexicon :
            other_ngram = ngram[:-1] + (word,)
            if self.ngram_counts[len(other_ngram)-1][other_ngram] == 0 :
                normalizer += self.backoff_prob(other_ngram[:-1], beta)
        
        return alpha * self.backoff_prob(ngram[:-1], beta) / normalizer
        


training_file = "./training/formatted_training/test_training.txt"
with open(training_file, 'r') as file :
    corpus = [line.rstrip('\n') for line in file]

model = Ngram_model(corpus, 3, 0)
print(len(model.lexicon))
print(model.backoff_prob(('in', 'scuffles'), 0.5))