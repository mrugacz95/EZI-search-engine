from argparse import ArgumentParser
from collections import defaultdict

import numpy as np

from tf_idf import load_data, similarity

KEYWORDS = 'data/keywords-2.txt'
STOP_WORDS = 'data/stop_words.txt'
DOCUMENTS = 'data/documents-2.txt'


def clustering(words_matrix, k=9, max_iter=5):
    items_num = words_matrix.shape[0]
    assignment = np.empty(items_num)  # which row is in which centroid

    # init with random items
    centroids = np.random.choice(items_num, size=k, replace=False)
    centroids = words_matrix[centroids, :]  # k random clusters, with length of keywords number

    iter = 0
    while iter < max_iter:
        iter += 1
        changed = False
        # assign items to groups
        for row_id, row in enumerate(words_matrix):
            sim, _ = similarity(row, centroids)
            cluster_id = sim[0]

            if assignment[row_id] != cluster_id:
                changed = True
                assignment[row_id] = cluster_id
        if not changed:
            break
        # calculate new centroids
        for cluster_id in range(k):
            indexes = np.argwhere(assignment == cluster_id)
            centroids[cluster_id] = np.mean(words_matrix[indexes, :], axis=0)
    return assignment, iter


def main(max_iter, k):
    original_docs = open(DOCUMENTS).read().split('\n\n')
    keywords = open(KEYWORDS).read().split('\n')
    stop_words = open(STOP_WORDS).read().split('\n')
    words_matrix, _, _, _ = load_data(original_docs, keywords, stop_words)

    original_docs = np.array(original_docs)

    assignment, iterations = clustering(words_matrix, k, max_iter)
    print(f'Completed in {iterations} iterations')
    for cluster_id in range(k):
        indexes = np.argwhere(assignment == cluster_id)
        print(f'Cluster #{cluster_id}:')
        for row in original_docs[indexes]:
            title = row[0].split('\n')
            print(f"\t{title}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max_iter', default=20, required=False, type=int)
    parser.add_argument('--k', default=9, required=False, type=int)
    args = parser.parse_args()
    main(args.max_iter, args.k)
