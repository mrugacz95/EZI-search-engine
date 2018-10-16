import re
from argparse import ArgumentParser

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


KEYWORDS = 'keywords.txt'
STOP_WORDS = 'stop_words.txt'
DOCUMENTS = 'documents.txt'


def main(query):
    steamer = PorterStemmer()

    def steam(word):
        return steamer.stem(word, 0, len(word) - 1)

    keywords = open(KEYWORDS).read().split('\n')

    stop_words = open(STOP_WORDS).read().split('\n')

    orginal_docs = open(DOCUMENTS).read().split('\n\n')

    # tokenizing
    docs = list(map(tokenize, orginal_docs))
    keywords = [tokenize(word)[0] for word in keywords]
    query = tokenize(query)

    # stop words
    for stop_word in stop_words:
        docs = [[word for word in doc if word.lower() != stop_word] for doc in docs]
        query = [word for word in query if word.lower() != stop_word]

    # normalization
    docs = [[normalize(word) for word in doc] for doc in docs]
    keywords = list(map(normalize, keywords))
    query = list(map(normalize, query))

    # steamming
    docs = [[steam(word) for word in doc] for doc in docs]
    keywords = list(map(steam, keywords))
    query = list(map(steam, query))
    # filling matrix
    query_vector = np.zeros(len(keywords), dtype=float)
    words_matrix = np.zeros((len(docs), len(keywords)), dtype=float)
    for doc_id, doc in enumerate(docs):
        for word in doc:
            if word in keywords:
                words_matrix[doc_id, keywords.index(word)] += 1
    words_matrix /= np.max(words_matrix, axis=1)[:, None]
    for word in query:
        if word in keywords:
            query_vector[keywords.index(word)] += 1
    query_vector /= np.max(query_vector)

    # computing idf frequencies
    np.seterr(divide='ignore')
    ida_frequency = np.log(len(docs) / np.count_nonzero(words_matrix, axis=0))
    ida_frequency[ida_frequency == np.inf] = 0

    # computing idf representation
    query_vector *= ida_frequency
    words_matrix *= ida_frequency

    # computing similarity
    length = np.linalg.norm(query_vector) * np.linalg.norm(words_matrix, axis=1)
    length[length == 0] = 1
    sim = np.sum(query_vector * words_matrix, axis=1) / length

    # sim = [0 if np.isnan(s) else s for s in sim]
    results = np.argsort(sim)[::-1]

    # showing results
    for doc_idx in results:
        title = orginal_docs[doc_idx].split("\n")[0]
        print(f'{title:<80} {sim[doc_idx]:.2f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('query', help='Search query')
    args = parser.parse_args()
    main(args.query)
