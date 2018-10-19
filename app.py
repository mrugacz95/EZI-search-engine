import random
import re

import numpy as np
from nltk.corpus import wordnet

from steamer import PorterStemmer


def tokenize(doc):
    doc = re.sub(r'[\n\t ]+', ' ', doc)
    doc = re.sub(r'[^A-Za-z\d\s\-\']', ' ', doc)
    doc = re.sub(r' +', ' ', doc)
    return doc.split(' ')


def normalize(word):
    word = word.lower()
    return re.sub(r'[\-\']', '', word)


KEYWORDS = 'keywords.txt'
STOP_WORDS = 'stop_words.txt'
DOCUMENTS = 'documents.txt'

steamer = PorterStemmer()


def steam(word):
    return steamer.stem(word, 0, len(word) - 1)


def load_data(original_docs):
    keywords = open(KEYWORDS).read().split('\n')

    stop_words = open(STOP_WORDS).read().split('\n')
    # tokenizing
    docs = list(map(tokenize, original_docs))
    keywords = [tokenize(word)[0] for word in keywords]

    # stop words
    for stop_word in stop_words:
        docs = [[word for word in doc if word.lower() != stop_word] for doc in docs]

    # normalization
    docs = [[normalize(word) for word in doc] for doc in docs]
    keywords = set(map(normalize, keywords))

    # steamming
    docs = [[steam(word) for word in doc] for doc in docs]
    keywords = list(map(steam, keywords))
    # filling matrix
    words_matrix = np.zeros((len(docs), len(keywords)), dtype=float)
    for doc_id, doc in enumerate(docs):
        for word in doc:
            if word in keywords:
                words_matrix[doc_id, keywords.index(word)] += 1
    words_matrix /= np.max(words_matrix, axis=1)[:, None]

    # computing idf frequencies
    np.seterr(divide='ignore')
    ida_frequency = np.log(len(docs) / np.count_nonzero(words_matrix, axis=0))
    ida_frequency[ida_frequency == np.inf] = 0

    # computing idf representation
    words_matrix *= ida_frequency

    return words_matrix, ida_frequency, keywords, stop_words


def search(query, words_matrix, ida_frequency, keywords, stop_words):
    query = tokenize(query)
    query = list(map(normalize, query))
    normalized_query = query

    propositions = []
    for word in normalized_query:
        hyponyms = wordnet.synsets(word)[0].hyponyms()
        propositions.extend([lemma.name().replace('_', ' ') for synset in hyponyms for lemma in synset.lemmas()])
    if len(propositions) < 5:
        extensions = propositions
    else:
        extensions = random.choices(propositions, k=5)
    for stop_word in stop_words:
        query = [word for word in query if word.lower() != stop_word]
    query = list(map(steam, query))

    query_vector = np.zeros(len(keywords), dtype=float)
    for word in query:
        if word in keywords:
            query_vector[keywords.index(word)] += 1
    query_vector /= np.max(query_vector)
    query_vector[np.isnan(query_vector)] = 0

    query_vector *= ida_frequency

    # computing similarity
    length = np.linalg.norm(query_vector) * np.linalg.norm(words_matrix, axis=1)
    length[length == 0] = 1
    similarity = np.sum(query_vector * words_matrix, axis=1) / length

    # sim = [0 if np.isnan(s) else s for s in sim]
    return np.argsort(similarity)[::-1], similarity, extensions


def main():
    original_docs = open(DOCUMENTS).read().split('\n\n')
    data = load_data(original_docs)
    while True:
        query = input("Search: ")
        if query in ['stop', 'quit']:
            break
        results, similarity, extensions = search(query, *data)
        # showing results
        for doc_idx in results:
            title = original_docs[doc_idx].split("\n")[0]
            print(f'{title:<80} {similarity[doc_idx]:.2f}')
        if extensions:
            print('You can extend query with words:')
            print(', '.join(extensions))




if __name__ == '__main__':
    main()
