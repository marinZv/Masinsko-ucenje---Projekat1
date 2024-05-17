from nltk import FreqDist
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import math
import pandas as pd

porter = PorterStemmer()



# učitavanje csv fajla
data = pd.read_csv("disaster-tweets.csv")

# izdvajanje samo određene kolone po nazivu
all_data = data["text"]
print(all_data)

all_target=data["target"]



train_target=all_target[:int(0.8*len(all_target))]
test_target=all_target[int(0.8*len(all_target)):]
# konvertovanje u NumPy niz
corpus = all_data.to_numpy()

#print(corpus)

clean_corpus = []
stop_punc = set(stopwords.words('english')).union(set(punctuation))

nb=0

for doc in corpus:
  words = wordpunct_tokenize(doc)
  words_lower = [w.lower() for w in words]
  words_filtered = [w for w in words_lower if w not in stop_punc]
  words_stemmed = [porter.stem(w) for w in words_filtered]
  #print(nb)
  #print(doc)
  #print(words_stemmed)
  nb=nb+1
  clean_corpus.append(words_stemmed)


#print('Creating the vocab...')
vocab_set = set()
for doc in clean_corpus:
  for word in doc:
    vocab_set.add(word)
vocab = list(vocab_set)

print('Vocab:', list(zip(vocab, range(len(vocab)))))
print('Feature vector size: ', len(vocab))


def occ_score(word, doc):
   return 1 if word in doc else 0

def numocc_score(word, doc):
  return doc.count(word)

def freq_score(word, doc):
  return doc.count(word) / len(doc)


X_occ = np.zeros((len(clean_corpus), len(vocab)), dtype=np.float32)
for doc_idx in range(len(clean_corpus)):
  doc = clean_corpus[doc_idx]
  for word_idx in range(len(vocab)):
    word = vocab[word_idx]
    cnt = occ_score(word, doc)
    X_occ[doc_idx][word_idx] = cnt
#print('X_occ:')
#print(X_occ)


X_numocc = np.zeros((len(clean_corpus), len(vocab)), dtype=np.float32)
for doc_idx in range(len(clean_corpus)):
  doc = clean_corpus[doc_idx]
  for word_idx in range(len(vocab)):
    word = vocab[word_idx]
    cnt = numocc_score(word, doc)
    X_numocc[doc_idx][word_idx] = cnt
print('X_numocc:')
print(X_numocc)


X_freq = np.zeros((len(clean_corpus), len(vocab)), dtype=np.float32)
for doc_idx in range(len(clean_corpus)):
  doc = clean_corpus[doc_idx]
  for word_idx in range(len(vocab)):
    word = vocab[word_idx]
    cnt = freq_score(word, doc)
    X_freq[doc_idx][word_idx] = cnt
#print('X_freq:')
#print(X_freq)


class_names = ['Fake','Disasters']

#X_occ=np.asarray(X_occ)
num_rows=X_occ.shape[0]

X_train=X_occ[:int(0.8*num_rows)]
X_test=X_occ[int(0.8*num_rows):]


Y = np.asarray(train_target)
X= np.asarray(X_train)
test_bow=np.asarray(X_test)

Y_target=np.asarray(test_target)

#test_bow1=test_bow.flatten()
#print('TestBow1 ', test_bow)

#print('TestBow ', test_bow)
#print('X', X)



class MultinomialNaiveBayes:
  def __init__(self, nb_classes, nb_words, pseudocount):
    self.nb_classes = nb_classes
    self.nb_words = nb_words
    self.pseudocount = pseudocount

  def fit(self, X, Y):
    nb_examples = X.shape[0]

    # Racunamo P(Klasa) - priors
    # np.bincount nam za datu listu vraca broj pojavljivanja svakog celog
    # broja u intervalu [0, maksimalni broj u listi]
    self.priors = np.bincount(Y) / nb_examples
    #print('Priors:')
    #print(self.priors)

    # Racunamo broj pojavljivanja svake reci u svakoj klasi
    occs = np.zeros((self.nb_classes, self.nb_words))
    for i in range(nb_examples):#prolazak kroz sve tekstove
      c = Y[i]#klasa kako je taj tekst klasifikovan
      for w in range(self.nb_words):#kroz sve reci
        cnt = X[i][w]#uzimamo iz bagOfWords modela broj poj reci u tom tekstu
        occs[c][w] += cnt
    #print('Occurences:')
    #print(occs)

    # Racunamo P(Rec_i|Klasa) - likelihoods
    self.like = np.zeros((self.nb_classes, self.nb_words))
    for c in range(self.nb_classes):
      for w in range(self.nb_words):
        up = occs[c][w] + self.pseudocount
        down = np.sum(occs[c]) + self.nb_words*self.pseudocount
        self.like[c][w] = up / down
    #print('Likelihoods:')
    #print(self.like)

  def predict(self, bow):
    # Racunamo P(Klasa|bow) za svaku klasu
    probs = np.zeros(self.nb_classes)
    for c in range(self.nb_classes):
      prob = np.log(self.priors[c])
      for w in range(self.nb_words):
        cnt = bow[w]
        prob += cnt * np.log(self.like[c][w])
      probs[c] = prob
    # Trazimo klasu sa najvecom verovatnocom
    #print('\"Probabilites\" for a test BoW (with log):')
    #print(probs)
    prediction = np.argmax(probs)
    return prediction

  def predict_multiply(self, bow):
    probs = np.zeros(self.nb_classes)
    for c in range(self.nb_classes):
      prob = self.priors[c]
      for w in range(self.nb_words):
        cnt = bow[w]
        prob *= self.like[c][w] ** cnt
      probs[c] = prob

    prediction = np.argmax(probs)
    return prediction


correct = 0
model = MultinomialNaiveBayes(nb_classes=2, nb_words=X_occ.shape[1],
pseudocount=1)

model.fit(X, Y)

for i in range(test_bow.shape[0]):
  prediction = model.predict(test_bow[i])
  if prediction==Y_target[i]:
    correct = correct + 1


print(correct*100/test_bow.shape[0])


positive=np.zeros(X_numocc.shape[1])
negative=np.zeros(X_numocc.shape[1])


for i in range(X_numocc.shape[0]):
  if all_target[i]==1:
    for j in range(X_numocc.shape[1]):
      negative[j]=negative[j]+X_numocc[i][j]
  else:
    for j in range(X_numocc.shape[1]):
      positive[j]=positive[j]+X_numocc[i][j]

print("Negative")
print(negative)

print("Positive")
print(positive)


positive_indx=[]
negative_indx=[]


positive_max=[]
negative_max=[]

for i in range(5):
  positive_indx.append(np.argmax(positive))
  positive_max.append(positive[np.argmax(positive)])
  positive[np.argmax(positive)]=-1
  negative_indx.append(np.argmax(negative))
  negative_max.append(negative[np.argmax(negative)])
  negative[np.argmax(negative)]=-1


positive_words=[]
negative_words=[]

for i in range(5):
  positive[positive_indx[i]]=positive_max[i]
  negative[negative_indx[i]]=negative_max[i]
  positive_words.append(vocab[positive_indx[i]])
  negative_words.append(vocab[negative_indx[i]])



print("Pozitivne reci: ")
print(positive_words)

print("Negativne reci: ",negative_words )

indx=[]



for i in range(len(negative)):
  if negative[i]>=10 and positive[i]>=10:
    indx.append(i)

# LR(reč) = br. poj. u poz. tvitovima (reč) / br. poj. u neg. tvitovima (reč)
print("INDX ",indx)

LR_niz=[]

for i in range(len(indx)):
  LR_niz.append((positive[indx[i]] / negative[indx[i]],indx[i]))

LR_sortiran_niz = sorted(LR_niz,key=lambda x:x[0], reverse=True)
# print(LR_sortiran_niz)

LR_max = LR_sortiran_niz[:5]
for i in range(len(LR_max)):
  print(vocab[LR_max[i][1]])
# print(LR_max)
LR_min = LR_sortiran_niz[-5:]
for i in range(len(LR_min)):
  print(vocab[LR_min[i][1]])
# print(LR_min)