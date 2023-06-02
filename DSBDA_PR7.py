from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import math
import nltk

nltk.download('punkt')

text = """Hello Mr. Smith, how are you doing today? The weather is
great, and city is awesome. The sky is pinkish-blue. You shouldn't
eat cardboard"""

tokenized_text = sent_tokenize(text)
print(tokenized_text)

tokenized_word = word_tokenize(text)
print(tokenized_word)

fdist = FreqDist(tokenized_word)
print(fdist)

fdist.most_common(2)
fdist.plot(30, cumulative=False)
plt.show()

sent = "Albert Einstein was born in Ulm, Germany in 1879."
tokens = nltk.word_tokenize(sent)
print(tokens)

nltk.download('averaged_perceptron_tagger')
nltk.pos_tag(tokens)
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
print(stop_words)

filtered_sent = []
for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)

print("Filterd Sentence:", filtered_sent)
print("Tokenized Sentence:", tokenized_word)

ps = PorterStemmer()
stemmed_words = []

for w in filtered_sent:
    stemmed_words.append(ps.stem(w))

print("Filtered Sentence:", filtered_sent)
print("Stemmed Sentence:", stemmed_words)


nltk.download('wordnet')
lem = WordNetLemmatizer()
stem = PorterStemmer()
word = "flying"
print("Lemmatized Word:", lem.lemmatize(word, "v"))
print("Stemmed Word:", stem.stem(word))


first_sentence = "Data Science is the sexiest job of the 21st century"
# split so each word have their own string
second_sentence = "machine learning is the key for data science"
first_sentence = first_sentence.split(" ")
# join them to remove common duplicate words
second_sentence = second_sentence.split(" ")
total = set(first_sentence).union(set(second_sentence))
print(total)

wordDictA = dict.fromkeys(total, 0)
wordDictB = dict.fromkeys(total, 0)
for word in first_sentence:
    wordDictA[word] += 1

for word in second_sentence:
    wordDictB[word] += 1


pd.DataFrame([wordDictA, wordDictB])


def computeTF(wordDict, doc):
    tfDict = {}
    corpusCount = len(doc)
    for word, count in wordDict.items():
        tfDict[word] = count/float(corpusCount)
    return (tfDict)  # running our sentences through the tffunction:


tfFirst = computeTF(wordDictA, first_sentence)
tfSecond = computeTF(wordDictB, second_sentence)  # Converting todataframe for
visualizationtf = pd.DataFrame([tfFirst, tfSecond])
visualizationtf


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
filtered_sentence = [w for w in wordDictA if not w in stop_words]
print(filtered_sentence)


def computeIDF(docList):
    idfDict = {}
    N = len(docList)

    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / (float(val) + 1))

    return (idfDict)  # inputing our sentences in the log file


idfs = computeIDF([wordDictA, wordDictB])


def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return (tfidf)


# running our two sentences through the IDF:
idfFirst = computeTFIDF(tfFirst, idfs)
idfSecond = computeTFIDF(tfSecond, idfs)

# putting it in a dataframe
idf = pd.DataFrame([idfFirst, idfSecond])
print(idf)

firstV = "Data Science is the sexiest job of the 21st century"
secondV = "machine learning is the key for data science"
# calling the TfidfVectorizer
vectorize = TfidfVectorizer()

response = vectorize.fit_transform([firstV, secondV])
print(response)
