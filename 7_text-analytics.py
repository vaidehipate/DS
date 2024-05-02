'''7) Text Analytics
1. Extract Sample document and apply following document preprocessing methods:
Tokenization, POS Tagging, stop words removal, Stemming and Lemmatization.
2. Create representation of documents by calculating Term Frequency and Inverse
DocumentFrequency.'''


import nltk
nltk.download('all')

# Tokenization using NLTK
from nltk import word_tokenize, sent_tokenize
text = "The sun shines brightly in the clear blue sky.Birds chirp melodiously as they flit from tree to tree."
sentences = sent_tokenize(text)
words= word_tokenize(text)
print(sentences)
print(words)

#Stemming
from nltk.stem import PorterStemmer

# create an object of class PorterStemmer
porter = PorterStemmer()
print(porter.stem("play"))
print(porter.stem("playing"))
print(porter.stem("plays"))
print(porter.stem("played"))

print(porter.stem("Communication"))

from nltk.stem import WordNetLemmatizer
# create an object of class WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("plays", 'v'))
print(lemmatizer.lemmatize("played", 'v'))
print(lemmatizer.lemmatize("play", 'v'))
print(lemmatizer.lemmatize("playing", 'v'))

'''
So, when you pass 'v' as the second argument to lemmatizer.lemmatize(),
you're instructing the lemmatiser to treat the input word as a verb
and return its base or dictionary form as a verb.
'''

print(lemmatizer.lemmatize("Communication", 'v'))

'''
 stemming focuses on removing suffixes to obtain a base form of words,
 lemmatisation aims to return the base or dictionary form of words, 
 taking into account their meaning and context.
 Lemmatisation generally produces more accurate results.
 '''

#POS Tagging

from nltk import pos_tag
from nltk import word_tokenize

text = "The sun shines brightly in the clear blue sky"
tokenized_text = word_tokenize(text)
tags = pos_tag(tokenized_text)
tags

#Explaination of output of pos tagging:
'''
Explanation of the POS tags:

'DT': Determiner (e.g., "the")
'NN': Noun, singular or mass (e.g., "sun", "sky")
'VBZ': Verb, 3rd person singular present (e.g., "shines")
'RB': Adverb (e.g., "brightly")
'IN': Preposition or subordinating conjunction (e.g., "in")
'JJ': Adjective (e.g., "clear", "blue")

'''


#Stopword removal

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
text = "This is an example sentence demonstrating stop word removal."
filtered_text = [word for word in word_tokenize(text) if word.lower() not in stop_words]
print(filtered_text)

import re

text = "This is a sentence with, punctuation!"
clean_text = re.sub(r'[^\w\s]', '', text)
print(clean_text)

#part 2

from sklearn.feature_extraction.text import TfidfVectorizer

# creation of documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the documents to calculate TF-IDF representation
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Convert TF-IDF matrix to array for easier manipulation
tfidf_array = tfidf_matrix.toarray()

# Get feature names (terms)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Print TF-IDF matrix (document-term matrix)
print("TF-IDF Matrix (Document-Term Matrix):")
print(tfidf_array)

# Print feature names (terms)
print("\nFeature Names (Terms):")
print(feature_names)


