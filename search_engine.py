import random

import numpy as np
from nltk.corpus import wordnet

from tf_idf import tokenize, normalize, load_data, steam, similarity

KEYWORDS = 'data/keywords.txt'
STOP_WORDS = 'data/stop_words.txt'
DOCUMENTS = 'data/documents.txt'


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
    result, sim = similarity(query_vector, words_matrix)
    return result, sim, extensions


def main():
    original_docs = open(DOCUMENTS).read().split('\n\n')
    keywords = open(KEYWORDS).read().split('\n')
    stop_words = open(STOP_WORDS).read().split('\n')
    data = load_data(original_docs, keywords, stop_words)
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
