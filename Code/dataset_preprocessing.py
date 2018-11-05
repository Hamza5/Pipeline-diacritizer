"""
Module containing several functions to read and correct several errors in Tashkeela dataset and to convert its data.
"""

import re
import numpy as np


# Hexadecimal values taken from https://www.unicode.org/charts/
D_NAMES = ['Fathatan', 'Dammatan', 'Kasratan', 'Fatha', 'Damma', 'Kasra', 'Shadda', 'Sukun']
NAME2DIACRITIC = dict((name, chr(code)) for name, code in zip(D_NAMES, range(0x064B, 0x0653)))
DIACRITIC2NAME = dict((code, name) for name, code in NAME2DIACRITIC.items())
ARABIC_DIACRITICS = frozenset(NAME2DIACRITIC.values())
ARABIC_LETTERS = frozenset([chr(x) for x in (list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B)))] + ['ـ'])
ARABIC_SYMBOLS = ARABIC_LETTERS | ARABIC_DIACRITICS
XML_TAG = r'(?:<.+>)+'
SENTENCE_SEPARATORS = '.:؟!'
SPACES = ' \t'
PUNCTUATION = SENTENCE_SEPARATORS + '۩﴿﴾«»؛،ـ' +\
              ''.join([chr(x) for x in range(0x0021, 0x0030)]+[chr(x) for x in range(0x003A, 0x0040)] +
                      [chr(x) for x in range(0x005B, 0x0060)]+[chr(x) for x in range(0x007B, 0x007F)])
DATETIME_REGEXP = re.compile(r'(?:\d+[-/:\s]+)+\d+')
NUMBER_REGEXP = re.compile(r'\d+(?:\.\d+)?')
DOTS_NO_URL = r'(?<!\w)(['+SENTENCE_SEPARATORS+']+)(?!\w)'
WORD_TOKENIZATION_REGEXP = re.compile('([' + ''.join(ARABIC_SYMBOLS) + ']+)')
SENTENCE_TOKENIZATION_REGEXP = re.compile(DOTS_NO_URL + '|' + XML_TAG)
CHAR2INDEX = dict((l, n) for n, l in enumerate(sorted(ARABIC_LETTERS - {'ـ'})))
CHAR2INDEX.update(dict((v, k) for k, v in enumerate([' ', '0'], len(CHAR2INDEX))))
INDEX2CHAR = dict((v, k) for k, v in CHAR2INDEX.items())


def clear_diacritics(text):
    """
    Remove all standard diacritics from the text, leaving the letters only.
    :param text: str, the diacritized text.
    :return: str, the text undiacritized.
    """
    assert isinstance(text, str)
    return ''.join([l for l in text if l not in ARABIC_DIACRITICS])


def keep_selected_diacritics(text, diacritics):
    """
    Remove only the standard diacritics which are not specified.
    :param text: str, the diacritized text.
    :param diacritics: set of str, diacritics to be kept.
    :return: the text without the diacritics that should be removed.
    """
    assert isinstance(text, str)
    assert isinstance(diacritics, set) and diacritics.issubset(ARABIC_DIACRITICS)
    return ''.join([l for l in text if l not in ARABIC_DIACRITICS - diacritics])


def extract_diacritics(text):
    """
    Return the diacritics from the text while keeping their original positions.
    :param text: str, the diacritized text.
    :return: list of str, the diacritics
    """
    assert isinstance(text, str)
    diacritics = []
    for i in range(1, len(text)):
        if text[i] in ARABIC_DIACRITICS:
            diacritics.append(text[i])
        elif text[i - 1] not in ARABIC_DIACRITICS:
            diacritics.append('')
    if text[-1] not in ARABIC_DIACRITICS:
        diacritics.append('')
    return diacritics


def merge_diacritics(undiacritized_text, diacritics):
    """
    Reconstruct the diacritized text from an undiacritized text and a list of corresponding diacritics.
    :param undiacritized_text: str, the undiacritized text.
    :param diacritics: list of str, the corresponding diacritics, as returned by extract_diacritics function.
    :return: str, the diacritized text.
    """
    assert isinstance(undiacritized_text, str)
    assert isinstance(diacritics, list) and set(diacritics).issubset(ARABIC_DIACRITICS.union(['']))
    i = 0
    j = 0
    sequence = []
    while i < len(undiacritized_text) and j < len(diacritics):
        sequence.append(undiacritized_text[i])
        i += 1
        if diacritics[j] in ARABIC_DIACRITICS:
            sequence.append(diacritics[j])
            if DIACRITIC2NAME[diacritics[j]] == 'Shadda' and j+1 < len(diacritics) and \
                    diacritics[j+1] in ARABIC_DIACRITICS - {diacritics[j]}:
                sequence.append(diacritics[j+1])
                j += 1
        j += 1
    return ''.join(sequence)


def fix_double_diacritics_error(diacritized_text):
    """
    Remove the duplicated diacritics by leaving the second one only when there are two incompatible diacritics.
    :param diacritized_text: the text containing the arabic letters with diacritics.
    :return: str, the fixed text.
    """
    assert isinstance(diacritized_text, str)
    fixed_text = diacritized_text[0]
    for x in diacritized_text[1:]:
        if x in ARABIC_DIACRITICS and fixed_text[-1] in ARABIC_DIACRITICS:
            if fixed_text[-1] != NAME2DIACRITIC['Shadda']:
                fixed_text = fixed_text[:-1]
        fixed_text += x
    return fixed_text


def clean_text(text):
    """
    Remove the unwanted characters from the text.
    :param text: str, the unclean text.
    :return: str, the cleaned text.
    """
    assert isinstance(text, str)
    # Clean HTML garbage, tatweel, dates, and replace numbers.
    return NUMBER_REGEXP.sub('0', DATETIME_REGEXP.sub('', text.replace('ـ', '').replace('&quot;', '')))


def tokenize(sentence):
    """
    Tokenize a sentence into a list of words.
    :param sentence: str, the sentence to be tokenized.
    :return: list of str, list containing the words.
    """
    assert isinstance(sentence, str)
    return list(filter(lambda x: x != '' and x.isprintable(), re.split(WORD_TOKENIZATION_REGEXP, sentence)))


def filter_tokenized_sentence(sentence, min_words=2, min_word_diac_rate=0.8):
    """
    Accept or void a sentence, and clean the tokens.
    :param sentence: the sentence to be filtered.
    :param min_words: minimum number of arabic words that must be left in the cleaned sentence in order to be accepted.
    :param min_word_diac_rate: rate of the diacritized words to the number of arabic words in the sentence.
    :return: list of str, the cleaned tokens or an empty list.
    """
    assert isinstance(sentence, list) and all(isinstance(w, str) for w in sentence)
    assert min_words >= 0
    assert min_word_diac_rate >= 0
    new_sentence = []
    if len(sentence) > 0:
        diac_word_count = 0
        arabic_word_count = 0
        for token in sentence:
            word_chars = set(token)
            if word_chars.issubset(ARABIC_SYMBOLS):
                arabic_word_count += 1
                if word_chars & ARABIC_DIACRITICS != set():
                    diac_word_count += 1
            token = token.strip(SPACES+PUNCTUATION)
            if token != '' and (set(token).issubset(ARABIC_SYMBOLS) or NUMBER_REGEXP.match(token)):
                new_sentence.append(token)
        if arabic_word_count >= min_words:
            if diac_word_count / arabic_word_count > min_word_diac_rate:
                return new_sentence
    return []


def read_text_file(file_path):
    """
    Reads a text file and returns a list of individual sentences.
    :param file_path: The path of the file.
    :return: list of str, each str is a sentence.
    """
    assert isinstance(file_path, str)
    sentences = []
    with open(file_path, 'rt', encoding='utf-8') as dataset_file:
        for line in dataset_file:
            line = clean_text(line.strip(SPACES+'\n'))
            if line == '' or not line.isprintable():
                continue
            fragments = [x.strip(SPACES) for x in
                         filter(lambda x: x != '', re.split(SENTENCE_TOKENIZATION_REGEXP, line)) if x is not None]
            if len(fragments) > 1:
                for f1, f2 in zip(fragments[:-1], fragments[1:]):
                    if f2 in SENTENCE_SEPARATORS:
                        sentences.append(f1+f2)
                    elif f1 in SENTENCE_SEPARATORS:
                        continue
                    else:
                        sentences.append(f1)
            else:
                sentences.extend(fragments)
    return sentences


def text_to_one_hot(text):
    """
    Transform a sentence into a one-hot matrix where the first dimension is the characters and the second is the index.
    :param text: str, the sentence to be transformed.
    :return: ndarray, 2D NumPy array representing the new format.
    """
    assert isinstance(text, str) and set(text) & ARABIC_DIACRITICS == set()
    char_vectors = np.zeros((len(text), len(CHAR2INDEX)))
    for i in range(len(text)):
        if text[i] in ARABIC_LETTERS - {'ـ'}:
            char_vectors[i, CHAR2INDEX[text[i]]] = 1
        elif text[i].isnumeric():
            char_vectors[i, CHAR2INDEX['0']] = 1
        elif text[i].isspace():
            char_vectors[i, CHAR2INDEX[' ']] = 1
    return char_vectors


def add_time_steps(one_hot_matrix, time_steps):
    """
    Transform a 2D one-hot matrix into a 3D one containing time steps.
    :param one_hot_matrix: ndarray, the one-hot matrix
    :param time_steps: int, the number of time steps
    :return: ndarray, 3D matrix with time steps as a second dimension.
    """
    assert isinstance(one_hot_matrix, np.ndarray) and len(one_hot_matrix.shape) == 2
    assert isinstance(time_steps, int) and time_steps > 1
    X = np.empty((one_hot_matrix.shape[0], time_steps, one_hot_matrix.shape[1]))
    padded_one_hot = np.concatenate((np.zeros((time_steps-1, one_hot_matrix.shape[1])), one_hot_matrix))
    for i in range(X.shape[0]):
        X[i] = padded_one_hot[i:i+time_steps]
    return X


def input_to_sentence(batch):
    assert isinstance(batch, np.ndarray) and len(batch.shape) == 3
    one_hot = batch[:, -1]
    indices = np.argmax(one_hot, axis=1)
    return ''.join([INDEX2CHAR[i] for i in indices])
