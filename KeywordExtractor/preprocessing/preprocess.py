import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from stopwords import STOPWORDS
import string

"""
    Requires the Punkt Tokenizer Models from NLTK for word_tokenize
    python interpreter:
        >>> import nltk
        >>> nltk.download('punkt')
    
    The POS tagging requires the Averaged Perceptron Tagger to access
    nltk.pos_tag(...)
    Installed using
        >>> import nltk
        >>> nltk.download('averaged_perceptron_tagger')
    
    Lemmatization needs the Wordnet Lemmatizer
        >>> import nltk
        >>> nltk.download('wordnet')
"""

filter_STP = set(STOPWORDS)
REQD_STP = ["NN", "NNS", "NNP", "NNPS", "FW"]
lemmatizer = WordNetLemmatizer()


def tokenize_text(document):
    """
    Tokenizes a text document and strips it off of its punctuations
    :param document: string containing the document
    :return: tokenized document
    """
    document = document.lower()
    document = [x for x in document if x in string.printable]
    processed = ''.join(document)
    tokenized_doc = [word_tokenize(sent) for sent in sent_tokenize(processed)]
    return tokenized_doc


def get_stopwords(document):
    """
    Generates the relevant stopwords for the given word and its POS
    :param document: list of list of tuples of word and its POS for each sentence in document
    :return: list of stopwords
    """
    stp = set()
    for sent in document:
        stp = stp | set([word for word, pos in sent if pos not in REQD_STP])

    return stp | filter_STP | set(list(string.punctuation))


def get_pos_tagged_doc(tokenized_text):
    """
    Given a tokenized document, return list of list of 2-tuples in a sentence
    consisting of the word and it's POS
    :param tokenized_text: list of list of words in each sentence of document
    :return: list of list of tuples of word and its POS for each sentence in document
    """
    pos_tagged = [nltk.pos_tag(tokenized_sent) for tokenized_sent in tokenized_text]
    return pos_tagged


def lemmatize_doc(pos_tagged_text):
    """
    given a POS tagged text document, return the lemmatized document
    without the POS tags
    :param pos_tagged_text: list of list of tuples containing word and its
                            POS for each sentence in the document
    :return: List of list of words for each sentence in docuement after lemmatization
    """
    def lemmatized_word(tup):
        curr_tag = wordnet.NOUN
        word, tag = tup
        if tag.startswith('V'):
            curr_tag = wordnet.VERB
        elif tag.startswith('J'):
            curr_tag = wordnet.ADJ
        elif tag.startswith('R'):
            curr_tag = wordnet.ADV

        if tag.startswith('J'):
            curr_tag = 'a'
        else:
            curr_tag = 'n'

        return lemmatizer.lemmatize(word, curr_tag)

    lemmatized_text = []
    for sent in pos_tagged_text:
        lemmatized_text.append([lemmatized_word(tup) for tup in sent])

    return lemmatized_text


def filter_stopwords(tokenized_text, doc_stopwords):
    """
    strip the document of its stopwords
    :param tokenized_text: list of list of words of each sentence in doc
    :param doc_stopwords: set consisting the stopwords
    :return: final document stripped of the stopwords
    """

    cleaned_text = []
    for sent in tokenized_text:
        processed_sent = [word for word in sent if word not in doc_stopwords]
        cleaned_text.append(processed_sent)

    return cleaned_text


def retrieve_phrases(lemmatized_text, stopwords):
    """
    Lemmatized text is split into list of phrases
    :param lemmatized_text: Lemmatized text
    :param stopwords: set of stopwords
    :return: list of list consisting of phrases
    """

    phs = []
    for sentence in lemmatized_text:
        ph = []
        for word in sentence:
            if word not in stopwords:
                ph.append(word)
            else:
                if ph:
                    phs.append(ph)
                ph = []

    return [list(x) for x in set(tuple(x) for x in phs)]


# Pipeline
def get_cleaned_text(doc):
    """
    Returns the text containing important words and stripped off of punctuations and the phrases
    :param doc: string consisting of the document
    :return: list of list of words of each sentence in the document and the considered phrases
    """
    doc = tokenize_text(doc)
    doc = get_pos_tagged_doc(doc)
    lemmatized_doc = lemmatize_doc(doc)
    doc = get_pos_tagged_doc(lemmatized_doc)
    stopwords = get_stopwords(doc)
    phrases = retrieve_phrases(lemmatized_doc, stopwords)
    doc = filter_stopwords(lemmatized_doc, stopwords)
    return doc, phrases
