# Search engine

### Description

Repository contains script for searching documents with provided phrase. Algorithm is based on TF-IDF, Porter Stemming Algorithm, tokenizing and NLTK WordNet.

It also contains k-means for clustering document based on topic.

### Example usage

```bash
python3 search_engine.py
Search: computing
Machine Learning at UC Santa Cruz                                                0.44
Department of Computer Science                                                   0.43
The Hebrew University - School of Computer Science and ...                       0.34
Machine Learning Group                                                           0.26
Machine Learning textbook                                                        0.26
...
```

```bash
python3 kmeans.py
Completed in 4 iterations
Cluster #0:
	['animal planet', 'Animal Planet :: Home Page... Jane Goodall, K9 Karma, Miami Animal Police, ', "      Mutual of Omaha's Wild Kingdom, New Breed Vets, Planet's Funniest Animals, ", '      The Crocodile Hunter, Whoa! ...']
	['animal planet', 'Desert Animals & Wildlife Lots of links to our many exciting and ', '      informative pages about the North American deserts and its animals.']
	...
Cluster #1:
	['svd', 'Singular value decomposition and determinants', 'Singular value decomposition and determinants. ... M, and calculates the ', 'singular', 'value decomposition of M. This consists of a matrix of orthonormal columns ...']
	['svd', 'Matrix Reference Manual: Matrix Decompositions', 'A=LDMH is an alternative singular value decomposition of A iff UHL = DIAG(Q1,', 'Q2, ..., Qk, R) and VHM = DIAG(Q1, Q2, ..., Qk, S) where Q1, Q2, ...']
	...
...
```


### Used libraries

```
numpy
nltk
```