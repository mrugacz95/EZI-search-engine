import re

import numpy as np

from steamer import PorterStemmer


def tokenize(doc):
    doc = re.sub(r'[\n\t ]+', ' ', doc)
    doc = re.sub(r'[^A-Za-z\d\s\-\']', ' ', doc)
    doc = re.sub(r' +', ' ', doc)
    return doc.split(' ')


def normalize(word):
    word = word.lower()
    return re.sub(r'[\-\']', '', word)


steamer = PorterStemmer()


def steam(word):
    return steamer.stem(word, 0, len(word) - 1)


def load_data(original_docs, keywords, stop_words):
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


def similarity(vector, matrix):
    length = np.linalg.norm(vector) * np.linalg.norm(matrix, axis=1)
    length[length == 0] = 1
    sim = np.sum(vector * matrix, axis=1) / length
    return np.argsort(sim)[::-1], sim
