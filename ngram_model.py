import nltk
from collections import defaultdict, Counter
import math

def get_lexicon(corpus, minimum) :
    """
    returns the set of words in a corpus with count greater
    than 1
    """
    counts = Counter(corpus)
    return set(word for word in counts if counts[word] > minimum)

def pre_process(corpus, lexicon, POS_TAG) :
    """
    pos tag corpus and replace words in corpus
    with UNK token if word not in lexicon
    """
    corpus = [nltk.pos_tag(line) for line in corpus]

    #replace words that have small counts with 'UNK' token
    #only if counting words and not tags
    if POS_TAG : return corpus

    for line in corpus :
        for i, (word, tag) in enumerate(line) :
            if word not in lexicon :
                line[i] = ('UNK', 'UNK')
    
    return corpus

def get_ngrams(line, n) :
    """
    returns list of ngrams given a list of tokens and ngram size n
    """
    if type(line) != list :
        print("ERR: did not get a list type when parsing ngrams.")
        print("Instead, we got ", type(line))
        print("It had value: ", line)
    bookended = [('START', 'START')] * max(1, n-1) + line + [('STOP', 'STOP')]
    return [tuple(bookended[i:i+n]) for i in range(len(bookended) - (n - 1))]

def count_words(corpus) :
    """
    returns a dictionary of dictionaries that word from tag emissions
    """
    word_counts = defaultdict(lambda: defaultdict(int))
    for line in corpus :
        for word, tag in line :
            word_counts[tag][word] += 1
        
    return word_counts
    

class Ngram_model :

    def __init__(self, corpus, n=3, POS_TAG=0) :
        """
        corpus : list of strings, each string a Shakespeare line
        n : size of ngram for model
        POS_TAG : 1 means use POS_TAGS for ngrams, 0 means use words
        """

        if n <= 0 :
            print("ERR: ngram size must be positive")
            exit(1)
        
        if POS_TAG != 0 and POS_TAG != 1 :
            print("ERR: POS_TAG argument must be 0 or 1")
            exit(1)
        
        self.n = n
        self.POS_TAG = POS_TAG
        #initialize for backoff calculations
        self.alpha = {}
        self.normalizer = {}

        corpus = [nltk.word_tokenize(line) for line in corpus]
        print("TRAINING CORPUS: TOKENIZED")

        self.lexicon = get_lexicon([word for line in corpus for word in line], 0)
        self.lexicon.add('START')
        self.lexicon.add('STOP')
        if not POS_TAG : self.lexicon.add('UNK')

        self.corpus = pre_process(corpus, self.lexicon, self.POS_TAG)
        print("TRAINING CORPUS: PREPROCESSED")
        
        #initialize total number of tokens
        self.M = 0
        #count up all the k-grams from unigram to ngram
        self.ngram_counts = [self.count_ngrams(i) for i in range(1, self.n+1)]
        print("TRAINING CORPUS: NGRAMS COUNTED")

        #count up word probabilites
        self.word_counts = count_words(self.corpus)

        #initialize beta for backoff prob
        self.beta = 0.2

        return
    
    def change_mode(self) :
        self.POS_TAG = abs(self.POS_TAG - 1)
    
    def get_ngrams(self, line, n) :
        ngrams = []
        for tuple_ngram in get_ngrams(line, n) :
            #pick either word or tag to count depending on instance of model
            ngrams.append(tuple([pair[self.POS_TAG] for pair in tuple_ngram]))
        return ngrams

    def count_ngrams(self, n) :
        ngram_counts = defaultdict(int)
        START = tuple(['START'] * n)
        for line in self.corpus :
            for ngram in self.get_ngrams(line, n) :
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

    def count(self, ng) :
        if len(ng) > self.n :
            print("ERR: ngram length out of range for model")
            exit(1)

        return self.ngram_counts[len(ng)-1][ng]
    
    def get_alpha(self, ng, beta) :
        if ng in self.alpha :
            return self.alpha[ng]
        
        #count how many words copmlete this n+1_gram
        num_grams = 0
        for word in self.lexicon :
            ng_plus1 = ng + (word,)
            if self.count(ng_plus1) > 0 :
                num_grams += 1
        
        return (num_grams * beta) / self.count(ng)
    
    def backoff_prob(self, ngram, beta=self.beta) :
        
        def get_normalizer(ng) :
            #avoid recalculating
            if ng in self.normalizer :
                return self.normalizer[ng]
            
            normalizer = 0
            #check to see if the larger n+1_gram has count of 0
            #if so, then add the probability of the ngram starting
            #from the first index of n+1_gram to the end to the normalizer
            for word in self.lexicon :
                if word == 'START' : continue
                ng_plus1 = ng + (word,)
                if self.count(ng_plus1) == 0 :
                    normalizer += self.backoff_prob(ng_plus1[1:], beta)

            self.normalizer[ng] = normalizer
            return self.normalizer[ng]
        
        #base case 1 : unigram
        #simple MLE
        if len(ngram) == 1:
            return self.count(ngram) / self.M
        
        ngram_count = self.count(ngram)
        #base case 2 : seen ngram
        #discounted MLE
        if ngram_count > 0 :
            return (ngram_count - beta) / self.count(ngram[:-1])
        
        #otherwise, we have to recurse
        alpha = self.get_alpha(ngram[:-1], beta)
        normalizer = get_normalizer(ngram[:-1])

        return alpha * self.backoff_prob(ngram[1:], beta) / normalizer
    
    def word_prob(self, word, tag) :
        if word == 'START' or word == 'STOP' :
            return 1
        
        #do laplacian smoothing
        alpha = 0.1
        tag_count = self.ngram_counts[0][(tag,)]
        voc_size = len(self.word_counts[tag])
        return (self.word_counts[tag][word] + alpha) / (tag_count + voc_size * alpha)
    
    def sentence_logprob(self, sentence, beta) :
        #sum ngram probs
        total = sum(math.log2(self.backoff_prob(ngram, beta)) for ngram in self.get_ngrams(sentence, self.n))
        #sum word probs, too, if we use POS tags for ngrams
        if self.POS_TAG : total += sum(math.log2(self.word_prob(word, tag)) for word, tag in sentence)
        return total

    def perplexity(self, corpus, beta):
        """
        calculate the log probability of an entire corpus
        assumes corpus is passed formatted or as line by line
        strings
        """
        #if we are passed an unformatted corpus
        if(type(corpus[0]) == str) :
            corpus = self.format_corpus(corpus)

        l, M = 0, 0
        for sentence in corpus :
            # add extra 1 to account for start/end tokens in padding trigrams
            # todo: FIX THIS TO ACCOUNT FOR size of N
            M += len(sentence) + 1
            if self.POS_TAG : M += len(sentence)
            l += self.sentence_logprob(sentence, beta)
        
        return 2 ** -(l/M)
    
    def format_corpus(self, corpus) :
        """
        format a corpus which is assumed to be a list of strings
        (each string a line) into the expectd format for the model
        """
        #preprocess the corpus
        corpus = [nltk.word_tokenize(line) for line in corpus]
        print("CORPUS TOKENIZED")
        corpus = pre_process(corpus, self.lexicon, self.POS_TAG)
        print("CORPUS POS TAGGED")
        return corpus
    
    def tune(self, tuning_corpus) :
        """
        learn beta parameter for katz backoff
        with tuning_corpus assumed to be list of strings
        (unformatted)
        """
        tuning_corpus = self.format_corpus(tuning_corpus)
        betas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.beta = max(betas, lambda beta : self.perplexity(tuning_corpus, beta))
        print("BETA VALUE LEARNED: ", self.beta)


def open_corpus(path) :
    with open(path, 'r') as file :
        corpus = [line.rstrip('\n') for line in file]
    return corpus

corpus = open_corpus("./training/formatted_training/training.txt")
model = Ngram_model(corpus, 3, 1)


# corpus = open_corpus("./training/formatted_training/test_training1.txt")
# model = Ngram_model(corpus, 2)
# beta = 0.5
# bigram = ('y', 'y')
# print(model.backoff_prob(('y', 'y'), beta))
# print(model.get_alpha(bigram[:-1], beta))
# print(model.perplexity(corpus, 0.9))


# val = model.backoff_prob(('START','x', 'y'),beta)
# print(val)

# corpus = open_corpus("./training/formatted_training/training.txt")







# model = Ngram_model(corpus, 3, 0)
# print(len(model.lexicon))
# print(model.backoff_prob(('in', 'scuffles'), 0.5))
# sentence = nltk.pos_tag(nltk.word_tokenize("Have glow'd like plated Mars, now bend, now turn,"))
# print(model.sentence_logprob(sentence, 0.0000000001))

# test_corpus = open_corpus("./test/formatted_test/test.txt")
# print(model.perplexity(test_corpus, 0.2))