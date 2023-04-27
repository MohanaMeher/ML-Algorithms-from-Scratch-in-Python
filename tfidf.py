import sys

import nltk
from nltk.stem.porter import *
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import xml.etree.cElementTree as ET
from collections import Counter
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import zipfile
import os

def gettext(xmltext) -> str:
    """
    Parses xmltext and returns the text from <title> and <text> tags
    """
    result = []
    xmltext = xmltext.encode('ascii', 'ignore') # ensure there are no weird char
    tree = ET.fromstring(xmltext)
    title = tree.find('.//title')
    result.append(title.text)
    for elem in tree.iterfind('.//text/*'):
        if elem.tag == 'p':
            result.append(elem.text)
    return ' '.join(result)

    

def tokenize(text) -> list:
    """
    Tokenizes text and returns a non-unique list of tokenized words
    found in the text. Normalizes to lowercase, strip punctuation,
    removes stop words, drops words of length < 3, strips digits.
    """
    text = text.lower()
    text = re.sub('[' + string.punctuation + '0-9\\r\\t\\n]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if len(w) > 2 and w not in ENGLISH_STOP_WORDS]  # ignore a, an, to, at, be, ...
    return tokens

def stemwords(words) -> list:
    """
    Reads a list of tokens/words, returns a new list with each word
    stemmed using a PorterStemmer.
    """
    ps = PorterStemmer()
    return [ps.stem(word) for word in words]


def tokenizer(text) -> list:
    return stemwords(tokenize(text))


def compute_tfidf(corpus:dict) -> TfidfVectorizer:
    """
    Creates and returns a TfidfVectorizer object after training it on
    the list of articles pulled from the corpus dictionary. Meaning,
    calls fit() on the list of document strings, figures out
    all the inverse document frequencies (IDF) for use later by
    the transform() function. The corpus argument is a dictionary
    mapping file name to xml text.
    """
    tfidf = TfidfVectorizer(input='content',
                        analyzer='word',
                        preprocessor=gettext,
                        tokenizer=tokenizer,
                        stop_words='english', # even more stop words
                        decode_error = 'ignore')
    return tfidf.fit(list(corpus.values()))


def summarize(tfidf:TfidfVectorizer, text:str, n:int):
    """
    Reads a trained TfidfVectorizer object and some XML text, returns
    up to n (word,score) pairs in a list. Discards any terms with
    scores < 0.09. Sorts the (word,score) pairs by TFIDF score in reverse order.
    """
    word_score_lst = []
    terms = tfidf.get_feature_names_out()
    result_matrix = tfidf.transform([text]).tocoo()
    for word_index, score in list(zip(result_matrix.col, result_matrix.data)):
        if score < 0.09:
            continue
        word_score_lst.append((terms[word_index], round(score, 3)))
    word_score_lst = sorted(word_score_lst, key=lambda x: x[1], reverse=True)
    return word_score_lst[:n]


def load_corpus(zipfilename:str) -> dict:
    """
    Reads a zip file containing root directory reuters-vol1-disk1-subset
    and a bunch of *.xml files, reads them from the zip file into
    a dictionary of (filename,xmltext) associations. Uses namelist() from
    ZipFile object to get list of xml files in that zip file.
    Converts filename reuters-vol1-disk1-subset/foo.xml to foo.xml
    as the keys in the dictionary. The values in the dictionary are the
    raw XML text from the various files.
    """
    result = {}
    with zipfile.ZipFile(zipfilename, 'r') as zf:
        filelist = zf.namelist()
        for filepath in filelist:
            filename = os.path.basename(filepath)
            if '.xml' not in filename:
                continue
            xml = zf.read(filepath)
            result[filename] = xml
    return result
