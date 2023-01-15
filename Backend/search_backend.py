# imports
import concurrent
import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby

import google
import pandas as pd
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from time import time
from timeit import timeit
from pathlib import Path
import pickle
import numpy as np
from google.cloud import storage
import itertools
import math
from inverted_index_gcp import *
from contextlib import closing
from collections import Counter
import gensim.models
import gensim.downloader as api

my_bucket_name = "project_bucket_sy"

title_path = 'title_index/postings_gcp_title_index/'
body_path = 'text_index/postings_gcp_text_index/'
anchor_path = 'anchor_index/postings_gcp_anchor_index/'


def read_pickle(bucket_name, pickle_route):
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(pickle_route)
    pick = pickle.loads(blob.download_as_string())
    return pick


##### read pkl from bucket #####

print('Start Loading')

print('Load Inverted Indexes')
inverted_title = read_pickle(my_bucket_name, "title_index/postings_title_gcp/title_index.pkl")
inverted_body = read_pickle(my_bucket_name, "text_index/postings_text_gcp/text_index.pkl")
inverted_anchor = read_pickle(my_bucket_name, "anchor_index/postings_anchor_gcp/anchor_index.pkl")

print('Load Pagerank & Pageview')
pagerank = read_pickle(my_bucket_name, "page_rank/pagerank_dict.pkl")
pageview = read_pickle(my_bucket_name, "page_view/pageview.pkl")

print('Load Helper Dictionaries')
doc_id_to_title_dic = read_pickle(my_bucket_name, "title_id_dict/doc_id_to_title_dict.pkl")
normalize_doc = read_pickle(my_bucket_name, "normalize_dict/normalize_doc_dict.pkl")

##### load word2vec #####

# download the model and return as object ready for use
print('Load Word2Vec')
model_glove_wiki = api.load("glove-wiki-gigaword-300")

print('Ready to use :)')


def similar_words(list_of_tokens, ecc):
    global model_glove_wiki
    candidates = model_glove_wiki.most_similar(positive=list_of_tokens, topn=4)
    res = [word for word, similarity in candidates if similarity > ecc]
    for tok in res:
        list_of_tokens += tokenize(tok)
    return list_of_tokens[:5]


def similar_words_long(list_of_tokens):
    global model_glove_wiki
    sim_words = []
    res = []

    for token in list_of_tokens:
        candidates = model_glove_wiki.most_similar(positive=token, topn=1)
        res += [(similarity, word) for word, similarity in candidates if similarity > 0.7]

    sim_words += list_of_tokens
    for sim, tok in sorted(res, reverse=True):
        if len(sim_words) < 5:
            sim_words += tokenize(tok)
        else:
            break

    return sim_words


##### corpus stopwords #####

nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)


##### tokenizer ######

def tokenize(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

    Parameters:
    -----------
    text: string , represting the text to tokenize.

    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    return list_of_tokens


##### helper functions #####

def title_string(doc_id):
    """
    returns the document title given its doc id
    """
    title = doc_id_to_title_dic.get(doc_id)
    return title if title else "Couldn't find matching title"


TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer


def read_posting_list(inverted, w, file_path) -> object:
    TUPLE_SIZE = 6
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        try:
            b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, file_path)
        except:
            print('couldnt use reader')

        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


def get_posting_list(idx, query, file_path):
    posting_lists = []
    for token in query:
        try:
            p = read_posting_list(idx, token, file_path)
        except:
            p = []
        posting_lists.append((token, p))
    return posting_lists


def q_normalize(q_counter):
    return 1 / math.sqrt(sum(count ** 2 for count in q_counter.values()))


def get_top_n(sim_dict, N=3):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores
    """

    return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[
           :N]


def calculate_similarity(search_q, idx, file_path, N=3):
    counter = Counter(tokenize(search_q))
    get_pls = get_posting_list(idx, list(counter.keys()), file_path)
    sim_dic = {}
    for token, p in get_pls:
        for doc_id, f in p:
            sim_dic[doc_id] = sim_dic.get(doc_id, 0) + counter[token] * f
    for d in sim_dic.keys():
        sim_dic[d] *= (q_normalize(counter)) * (normalize_doc[d])

    return get_top_n(sim_dic, N)


def binary_search(query, inverted_index, path, name):
    tokens = tokenize(query)
    posting_lists = []
    for token in tokens:
        try:
            posting_lists += read_posting_list(inverted_index, token, path)
        except:
            print(f"{token} was not found in inverted_{name}")
            pass

    if len(posting_lists) != 0:
        doc_ids = map(lambda x: x[0], posting_lists)
        counter = Counter(doc_ids)
        doc_title_tups = map(lambda x: (x[0], title_string(x[0])), counter.most_common())

        return list(doc_title_tups)

    return []  # in case no matching terms were fount in the inverted index return empty list


##### search ######


def search_title_helper(query):
    return binary_search(query, inverted_title, title_path, 'title')


def search_anchor_helper(query):
    return binary_search(query, inverted_anchor, anchor_path, 'anchor')


def search_body_helper(query):
    res = []
    try:
        cosine = calculate_similarity(query, inverted_body, body_path, N=100)
        cosine = [(x[0], title_string(x[0])) for x in cosine]
        res = cosine
    except Exception as e:
        print(f'Error in function: search_body_backend. Details: {e}')
    return res


##### pagerank & pageview #####

def page_rank_helper(wiki_ids):
    res = []
    for doc_id in wiki_ids:
        try:
            res.append(pagerank[doc_id])
        except:
            res.append(0)
    return res


def page_view_helper(lst):
    res = []
    for doc_id in lst:
        try:
            res.append(pageview[doc_id])
        except:
            res.append(0)
    return res


##### search using BM25 #####

class BM25_from_index:

    def __init__(self, index, file_path, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(index.DL)
        self.AVGDL = sum(index.DL.values()) / self.N
        self.file_path = file_path
        self.idf = {}

    def calc_idf(self, list_of_tokens):
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def _score(self, query, doc_id, pls_dict):
        score = 0.0
        doc_len = self.index.DL[doc_id]
        for term in query:
            if term in self.index.df.keys():
                if doc_id in pls_dict[term].keys():
                    freq = pls_dict[term][doc_id]
                    numerator = self.idf[term] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)
        return score

    def search(self, query, N=100):
        try:
            query_tokens = tokenize(query)

            if len(query) < 25:
                query = similar_words(query_tokens, 0.7)
            else:
                query = similar_words_long(query_tokens)

            idf = self.calc_idf(query)
            self.idf = idf

            d_pls = {}
            candidates = []

            for term in np.unique(query):
                if term in self.index.df.keys():
                    curr_lst = read_posting_list(self.index, term, self.file_path)
                    d_pls[term] = dict(curr_lst)

                    candidates += curr_lst

            candidates = set([x[0] for x in candidates])

            bm_25 = [(c, self._score(query, c, d_pls)) for c in candidates]
            bm_25 = sorted(bm_25, key=lambda x: x[1], reverse=True)[:N]

            return bm_25

        except:
            print(f'no mach for {query} in our search engine')
            pass


def merge_results(title_scores, body_scores, title_w=0.5, text_w=0.5):
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body).
     """

    temp_dict = defaultdict(float)
    for doc_id, score in title_scores:
        temp_dict[doc_id] += title_w * score
    for doc_id, score in body_scores:
        temp_dict[doc_id] += text_w * score

    return temp_dict


##### Set Wights #####

title_weight, text_weight = 0.5, 0.5

bm_weight = 0.5
page_view_weight = 0.25
page_rank_weight = 0.25

bm25_title = BM25_from_index(inverted_title, title_path)
bm25_body = BM25_from_index(inverted_body, body_path)


def merge_bm25_scores(query, bm25_title, bm25_body):
    """
    find the best bm25 docs and scores for the query.
    using title and body index
    """
    try:
        title_score = bm25_title.search(query, N=50)
        body_score = bm25_body.search(query, N=50)
        BM25_score = merge_results(title_score, body_score, title_weight, text_weight)
        # BM25_score = merge_results(title_score, body_score, t_s, b_s)
        return BM25_score
    except:
        print('An error occurred while searching')
        pass


def merge_bm25_pr_pv(bm25_dic):
    """
    calculate the final score using bm25, page view, page rank.
    """
    max_bm25 = max(bm25_dic.values())
    max_pr = max([pagerank[doc_id] for doc_id in bm25_dic.keys()])
    max_pv = max([pageview[doc_id] for doc_id in bm25_dic.keys()])

    for key, val in bm25_dic.items():
        bm = val
        page_rank = pagerank[key]
        page_view = pageview[key]
        bm25_dic[key] = (bm * bm_weight / max_bm25) + (page_rank * page_rank_weight / max_pr) + (page_view * page_view_weight / max_pv)
    return bm25_dic


def search_helper(query):
    try:
        if len(query) < 22:
            title_score = 0.4
            body_score = 1 - title_score
        else:
            title_score = 0.6
            body_score = 1 - title_score

        bm_25 = merge_bm25_scores(query, bm25_title=bm25_title, bm25_body=bm25_body)
        calc_scores = merge_bm25_pr_pv(bm_25)
        sort_res = list(sorted(calc_scores.items(), key=lambda x: x[1], reverse=True)[:10])
        res = [(doc_id, title_string(doc_id)) for doc_id, acc in sort_res]
        return list(res)

    except Exception as e:
        print(f'Error - {e}')
        return []
