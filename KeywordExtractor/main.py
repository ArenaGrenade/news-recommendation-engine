"""
Driver that returns list of potential keywords given the text document
paper: https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
"""

from preprocessing.preprocess import get_cleaned_text
from collections import OrderedDict
import numpy as np
import itertools
import math


def get_nodes(text):
    """
    Generates relevant nodes for textRank algorithm based on processed text
    :param text: processed text
    :return: set containing the nodes
    """
    nodes = set()
    for sent in text:
        nodes |= set(sent)
    return nodes


def get_node_pos(nodes):
    """
    Generate inverse position dictionary for each word in the set of nodes
    :param nodes: set containing the nodes
    :return: dictionary of nodes to index values
    """
    inv = OrderedDict()
    index = 0
    for word in nodes:
        inv[word] = index
        index += 1

    return inv


def get_graph(document, nodes, inv, window_size=3):
    """
    Generates edges between nodes
    :param window_size: Window size
    :param document: list of list of words for each sentence in document
    :param nodes: set of tokens
    :param inv: Dictionary mapping the token to its matrix location
    :return: adjacency matrix
    """
    n = len(nodes)
    graph = np.zeros((n, n))

    document = [word for sent in document for word in sent]

    for win_st in range(0, max(1, len(document) - window_size)):
        win_end = max(win_st + window_size, len(document))
        window_text = document[win_st:win_end]

        for (pos1, w1), (pos2, w2) in itertools.permutations(list(enumerate(window_text)), 2):
            i = inv[w1]
            j = inv[w2]

            if i != j:
                graph[i][j] += 1 / math.fabs(pos1 - pos2)

    return graph


def get_scores(nodes, inv, graph, iters=100, damping=0.7):
    """
    nodes are scored based on the formula
    :param iters: number of iterations
    :param graph: adjacency matrix
    :param nodes: set containing nodes
    :param inv: hash from word to its index
    :param damping: probability of jumping from a given vertex
                    to another random vertex in the graph
    :return: a list of scores of each word
    """
    threshold = 1e-3
    n = len(nodes)
    score = np.ones(n)
    ranks = np.zeros(n)
    for i in range(0, n):
        for j in range(0, n):
            ranks[i] += graph[i][j]

    for _ in range(0, iters):
        prev = score

        for i in range(0, n):
            sum_i = np.sum([(edge/ranks[j]) * score[j] for j, edge in enumerate(graph[i]) if edge != 0])
            score[i] = (1 - damping) + damping * sum_i

        if math.fabs(np.sum(prev - score)) <= threshold:
            break

    return score


def cleanify_phrases(phrases, nodes):
    """
    given list of phrases, remove unnecessary ones
    :param phrases: list of list of words in phrases
    :param nodes: set of words
    :return: list of candidate keywords
    """
    to_remove = set()
    phs = set([tuple(x) for x in phrases])
    for w in nodes:
        for ph in phs:
            if w in ph and tuple([w]) in phs and len(ph) > 1:
                to_remove.add(tuple([w]))
    return [list(t) for t in phs if t not in to_remove]


def get_phrase_scores(phrases, scores, nodes, inv):
    """
    given scores for each word, generate scores for the phrases
    :param phrases: list of list of words in each phrase
    :param scores: scores of each word in node
    :param nodes: set of words
    :param inv: hash from word to index
    :return: list containing scores for each phrase in the phrases list
    """

    phs_scores = dict()
    for ph in phrases:
        ph_score = np.average([scores[inv[w]] for w in ph])
        phs_scores[' '.join(ph)] = ph_score

    return phs_scores


def handle():
    with open('testData/data4.txt', 'r') as f:
        doc = f.read()
        doc, phrases = get_cleaned_text(doc)
        nodes = get_nodes(doc)
        inv = get_node_pos(nodes)
        graph = get_graph(doc, nodes, inv)
        scores = get_scores(nodes, inv, graph)
        candidate_keys = cleanify_phrases(phrases, nodes)
        print('candidates', candidate_keys)
        phrase_scores = get_phrase_scores(candidate_keys, scores, nodes, inv)
        print(dict(sorted(phrase_scores.items(), key=lambda item: item[1], reverse=True)))


handle()
