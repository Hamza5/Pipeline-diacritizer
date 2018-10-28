"""
Module containing several functions to read and correct several errors in Tashkeela dataset and to convert its data.
"""

import re

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
DOTS_NO_URL = r'(?<!\w)(['+SENTENCE_SEPARATORS+']+)(?!\w)'
WORD_TOKENIZATION_REGEXP = re.compile('([' + ''.join(ARABIC_SYMBOLS) + ']+)')
SENTENCE_TOKENIZATION_REGEXP = re.compile(DOTS_NO_URL + '|' + XML_TAG)


def clear_diacritics(text):
    """
    Remove all standard diacritics from the text, leaving the letters only.
    :param text: str, the diacritized text.
    :return: str, the text undiacritized.
    """
    assert isinstance(text, str)
    return ''.join([l for l in text if l not in ARABIC_DIACRITICS])


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
        elif text[i - 1] not in ARABIC_DIACRITICS or\
                text[i - 1] in ARABIC_DIACRITICS and DIACRITIC2NAME[text[i - 1]] == 'Shadda':
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
            if DIACRITIC2NAME[diacritics[j]] == 'Shadda' and diacritics[j+1] in ARABIC_DIACRITICS:
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
            if '&quot;' in token:  # HTML garbage
                token = token.replace('&quot;', '')
            if token != '':
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
            line = line.strip(SPACES+'\n')
            if line == '' or not line.isprintable():
                continue
            fragments = [x.strip(SPACES) for x in
                         filter(lambda x: x != '', re.split(SENTENCE_TOKENIZATION_REGEXP, line))]
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
