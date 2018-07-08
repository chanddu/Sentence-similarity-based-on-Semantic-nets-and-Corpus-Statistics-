# Sentence-Similarity-based-on-Semantic-Nets-and-Corpus-Statistics-
This is an implementation of the paper written by Yuhua Li, David McLean, Zuhair A. Bandar, James D. O’Shea, and Keeley Crockett. [Link]

The Sentence Similarity has been implemented as a linear combination of Semantic and Word order Similarity. Semantic and Word order Similarities are calculated from semantic and order vectors computed for each sentence with the assistance of wordnet.


## Modules Required
math<br>
os<br>
time<br>
sys<br>
numpy<br>
sklearn<br>
nltk<br>
- from nltk.corupus
  - wordnet
  - brown
  - stopwords

## Steps
1. Download the 2 main programs - similarity.py and main.py.
2. Construct the folder sub-structure as shown below:

[![Capture.png](https://s33.postimg.cc/9k8jg9167/Capture.png)](https://postimg.cc/image/q801iqvxn/)

3. similarity.py has all the main functions and will be called in main.py. Compile similarity.py first to make sure there are no errors. Then call the main.py
4. Put all the documents in text format to be compared for similarity.


[Link]: https://ieeexplore.ieee.org/document/1644735/
