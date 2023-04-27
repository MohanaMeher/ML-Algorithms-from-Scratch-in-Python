import sys
import re
import string
import os
import numpy as np
import codecs
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from typing import Sequence

import time

# From scikit learn that got words from:
# http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words
ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])


class defaultintdict(dict):
    def __init__(self):
        self._factory=int
        super().__init__()

    def __missing__(self, key):
        return 0


def filelist(root) -> Sequence[str]:
    """
    Returns a fully-qualified list of filenames under root directory; sorts names alphabetically.
    """
    allfiles = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            allfiles.append(os.path.join(path, name))
    return sorted(allfiles)


def get_text(filename:str) -> str:
    """
    Loads and returns the text of a text file, assuming latin-1 encoding
    """
    f = open(filename, encoding='latin-1', mode='r')
    s = f.read()
    f.close()
    return s


def words(text:str) -> Sequence[str]:
    """
    Reads a string, returns a list of words normalized as follows.
        Splits the string to make words first by using regex compile() function
        and string.punctuation + '0-9\\r\\t\\n]' to replace all those
        char with a space character.
        Splits on space to get word list.
        Ignores words < 3 char long.
        Lowercases all words
        Removes English stop words
    """
    ctrl_chars = '\x00-\x1f'
    regex = re.compile(r'[' + ctrl_chars + string.punctuation + '0-9\r\t\n]')
    nopunct = regex.sub(" ", text)  # delete stuff but leave at least a space to avoid clumping together
    words = nopunct.split(" ")
    words = [w for w in words if len(w) > 2]  # ignore a, an, to, at, be, ...
    words = [w.lower() for w in words]
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return words


def load_docs(docs_dirname:str) -> Sequence[Sequence]:
    """
    Loads all .txt files under docs_dirname and returns a list of word lists, one per doc.
    Also ignores empty and non ".txt" files.
    """
    docs = []
    file_list = filelist(docs_dirname)
    for file in file_list:
        if '.txt' in file and file != '':
            docs.append(words(get_text(file)))
    return docs


def vocab(neg:Sequence[Sequence], pos:Sequence[Sequence]) -> dict:
    """
    Reads neg and pos lists of word lists, constructs a mapping from word to word index,
    i.e. creates a dictionary using defaultintdict that maps keys (words) to values (index).

    Sorts the unique words in the vocab alphabetically so we standardize which
    word is associated with which word vector index.

    E.g., given neg = [['hi']] and pos=[['mom']], return:

    V = {'__unknown__':0, 'hi':1, 'mom:2}

    and so |V| is 3
    """
    V = defaultintdict()
    n = set(np.concatenate(neg))
    p = set(np.concatenate(pos))
    v_list = sorted(n.union(p))
    v_list.insert(0, '__unknown__')
    for i in range(len(v_list)):
        V[v_list[i]] = i
    return V


def vectorize(V:dict, docwords:Sequence) -> np.ndarray:
    """
    Returns a row vector (based upon V) for docwords with the word counts. 
    The first element of the
    returned vector is the count of unknown words. So |V| is |uniquewords|+1.
    """
    vec = np.zeros((len(V)))
    for word in docwords:
        vec[V[word]] += 1
    return vec


def vectorize_docs(docs:Sequence, V:dict) -> np.ndarray:
    """
    Returns a matrix where each row represents a documents word vector.
    Each column represents a single word feature. There are |V|+1
    columns because we leave an extra one for the unknown word in position 0.
    Invokes vector(V,docwords) to vectorize each doc for each row of matrix
    :param docs: list of word lists, one per doc
    :param V: Mapping from word to index; e.g., first word -> index 1
    :return: numpy 2D matrix with word counts per doc: ndocs x nwords
    """
    D = np.empty((len(docs), len(V)))
    for i in range(len(docs)):
        D[i] = vectorize(V, docs[i])
    return D


class NaiveBayes621:
    """
    This object behaves like a sklearn model with fit(X,y) and predict(X) functions.
    Limited to two classes, 0 and 1 in the y target.
    """
    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """
        Reads 2D word vector matrix X, one row per document, and 1D binary vector y
        train a Naive Bayes classifier. Two things are estimated, the prior p(c)
        and the likelihood P(w|c). P(w|c) is estimated by
        the number of times w occurs in all documents of class c divided by the
        total words in class c. p(c) is estimated by the number of documents
        in c divided by the total number of documents.

        The first column of X is a column of zeros to represent missing vocab words.
        """
        p_0 = np.log(len(np.where(y==0)) / len(y))
        p_1 = np.log(len(np.where(y==1)) / len(y))
        wcs_0 = np.array([0] * len(X[0]))
        wcs_1 = np.array([0] * len(X[0]))
        for ind in range(len(y)):
            if y[ind] == 0:
                wcs_0 = np.add(wcs_0, X[ind])
            else:
                wcs_1 = np.add(wcs_1, X[ind])
        wc_0 = np.sum(wcs_0)
        wc_1 = np.sum(wcs_1)
        den_0 = wc_0 + len(X[0])
        den_1 = wc_1 + len(X[0])
        prob_0 = lambda x: np.log((x + 1) / den_0)
        prob_func_0 = np.vectorize(prob_0)
        prob_1 = lambda x: np.log((x + 1) / den_1)
        prob_func_1 = np.vectorize(prob_1)
        wcs_0 = prob_func_0(wcs_0)
        wcs_1 = prob_func_1(wcs_1)
        self.pc = (p_0, p_1)
        self.pwc = (wcs_0, wcs_1)
        


    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        Reads 2D word vector matrix X, one row per document, returns binary vector
        indicating class 0 or 1 for each row of X.
        """
        res = []
        for x in X:
            c_0 = self.pc[0] + np.dot(x, self.pwc[0])
            c_1 = self.pc[1] + np.dot(x, self.pwc[1])
            if c_0 > c_1:
                res.append(0)
            else: 
                res.append(1)
        return res

def kfold_CV(model, X:np.ndarray, y:np.ndarray, k=4) -> np.ndarray:
    """
    Runs k-fold cross validation using model and 2D word vector matrix X and binary
    y class vector. Returns a 1D numpy vector of length k with the accuracies, the
    ratios of correctly-identified documents to the total number of documents. 
    """
    accuracies = []
    kf = KFold(n_splits=k, random_state=999, shuffle=True)
    for train_index, val_index in kf.split(X):
        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        accuracies.append(np.sum(y_val==y_pred) / len(y_val))
    return np.array(accuracies)