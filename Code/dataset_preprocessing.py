"""
Module containing several functions to read and correct several errors in Tashkeela dataset and converting its data.
"""

import re

# Hexadecimal values taken from https://www.unicode.org/charts/PDF/U0600.pdf
D_NAMES = ['Fathatan', 'Dammatan', 'Kasratan', 'Fatha', 'Damma', 'Kasra', 'Shadda', 'Sukun']
NAME2DIACRITIC = dict((name, chr(code)) for name, code in zip(D_NAMES, range(0x064B, 0x0653)))
DIACRITIC2NAME = dict((code, name) for name, code in NAME2DIACRITIC.items())
ARABIC_DIACRITICS = frozenset(NAME2DIACRITIC.values())
ARABIC_LETTERS = frozenset(chr(x) for x in (list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B))))
ARABIC_SYMBOLS = ARABIC_LETTERS | ARABIC_DIACRITICS
PUNCTUATION = '،؟ـ'+''.join([chr(x) for x in range(0x0021, 0x0030)]+[chr(x) for x in range(0x003A, 0x0040)] +
                            [chr(x) for x in range(0x005B, 0x0060)]+[chr(x) for x in range(0x007B, 0x007F)])
WORD_TOKENIZATION_REGEXP = re.compile(r'[\s' + PUNCTUATION + ']+')
SENTENCE_TOKENIZATION_REGEXP = re.compile(r'[.:\n]')


def clear_diacritics(word):
    """
    Removes all standard diacritics from a word, leaving the letters only.
    :param word: str, the diacritized word.
    :return: str, the word undiacritized.
    """
    assert isinstance(word, str) and set(word).issubset(ARABIC_SYMBOLS)
    return ''.join([l for l in word if l in ARABIC_LETTERS])


def extract_diacritics(word):
    """
    Returns the diacritics from the word while keeping their original positions.
    :param word: str, the diacritized word.
    :return: list of str, the diacritics
    """
    assert isinstance(word, str) and set(word).issubset(ARABIC_SYMBOLS)
    diacritics = []
    for i in range(1, len(word)):
        if word[i] in ARABIC_DIACRITICS:
            diacritics.append(word[i])
        elif word[i-1] in ARABIC_LETTERS or DIACRITIC2NAME[word[i-1]] == 'Shadda':
            diacritics.append('')
    if word[-1] in ARABIC_LETTERS:
        diacritics.append('')
    return diacritics


def merge_diacritics(undiacritized_word, diacritics):
    """
    Reconstructs the diacritized word from an undiacritized word and a list of corresponding diacritics.
    :param undiacritized_word: str, the undiacritized word.
    :param diacritics: list of str, the corresponding diacritics, as returned by extract_diacritics function.
    :return: str, the diacritized word.
    """
    assert isinstance(undiacritized_word, str) and set(undiacritized_word).issubset(ARABIC_LETTERS)
    assert isinstance(diacritics, list) and set(diacritics).issubset(ARABIC_DIACRITICS.union(['']))
    i = 0
    j = 0
    sequence = []
    while i < len(undiacritized_word) and j < len(diacritics):
        sequence.append(undiacritized_word[i])
        i += 1
        if diacritics[j] in ARABIC_DIACRITICS:
            sequence.append(diacritics[j])
            if DIACRITIC2NAME[diacritics[j]] == 'Shadda' and diacritics[j+1] in ARABIC_DIACRITICS:
                sequence.append(diacritics[j+1])
                j += 1
        j += 1
    return ''.join(sequence)


def tokenize(sentence):
    """
    Tokenize a sentence into a list of words.
    :param sentence: str, the sentence to be tokenized.
    :return: list of str, list containing the words.
    """
    assert isinstance(sentence, str)
    return list(filter(lambda x: x != '' and set(x).issubset(ARABIC_SYMBOLS) or x.isalnum(),
                       re.split(WORD_TOKENIZATION_REGEXP, sentence)))


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
            new_sentences = [s.strip(' \n') for s in re.split(SENTENCE_TOKENIZATION_REGEXP, line)]
            sentences.extend([s for s in new_sentences if s != ''])
    return sentences
