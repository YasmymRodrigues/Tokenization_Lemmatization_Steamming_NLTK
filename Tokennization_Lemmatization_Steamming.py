import nltk

text = "On Wednesday, the Association for Computing Machinery, the world’s largest society of computing professionals, announced that Hinton, LeCun and Bengio had won this year’s Turing Award for their work on neural networks. The Turing Award, which was introduced in 1966, is often called the Nobel Prize of computing, and it includes a $1 million prize, which the three scientists will share."

nltk.download('punkt')

#Sentence Tokenizer

from nltk.tokenize import sent_tokenize
sent_tk = sent_tokenize(text)
print("Sentence tokenizing the text: \n")
print(sent_tk)

#Word Tokenizer

from nltk.tokenize import word_tokenize
word_tk = word_tokenize(text)
print("Word tokenizing the text: \n")
print(word_tk)

#Removing stop words

nltk.download('stopwords')

from nltk.corpus import stopwords
sw = set(stopwords.words("english"))
print("Stop words in English language are: \n")
print(sw)

#filter out stop words
filtered_stops = [w for w in word_tk if not w in sw]
print("Text after removing stop words \n")
print(filtered_stops)

#Stemming

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

port_stem = PorterStemmer()
stemmed_words = []
for w in filtered_stops:
  stemmed_words.append(port_stem.stem(w))
print("Filtered Sentence: \n", filtered_stops, "\n")
print("Stemmed Sentece: \n", stemmed_words)

#Lemmatizing

import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

lemm_words = []

for i in range(50):
  lemm_words.append(lem.lemmatize(filtered_stops[i]))
print(lemm_words)

#Parts of Speech Tagging

nltk.download('averaged_perceptron_tagger')

from nltk import pos_tag #tag with part of speach
pos_tagged_words = pos_tag(word_tk)
print(pos_tagged_words)

#Frequency Distribution Plots

from nltk.probability import FreqDist
fd = FreqDist(word_tk)
print(fd)

import matplotlib.pyplot as plt
fd.plot(30, cumulative=False)
plt.show()

fd_alpha = FreqDist(text)
print(fd_alpha)
fd_alpha.plot(30, cumulative = False)