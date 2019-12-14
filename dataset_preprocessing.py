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
ARABIC_LETTERS = frozenset([chr(x) for x in (list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B)))])
ARABIC_SYMBOLS = ARABIC_LETTERS | ARABIC_DIACRITICS
EXTRA_SUKUN_REGEXP = re.compile(r'(?<=ال)' + NAME2DIACRITIC['Sukun'])
# YA_REGEXP = re.compile(r'ى(?=['+''.join(ARABIC_DIACRITICS)+r'])')
DIACRITIC_SHADDA_REGEXP = re.compile('(['+''.join(ARABIC_DIACRITICS)+'])('+NAME2DIACRITIC['Shadda']+')')
XML_TAG = r'(?:<.+?>)'
SENTENCE_SEPARATORS = ';,،؛.:؟!'
SPACES = ' \t'
PUNCTUATION = SENTENCE_SEPARATORS + '۩﴿﴾«»ـ' +\
              ''.join([chr(x) for x in range(0x0021, 0x0030)]+[chr(x) for x in range(0x003A, 0x0040)] +
                      [chr(x) for x in range(0x005B, 0x0060)]+[chr(x) for x in range(0x007B, 0x007F)])
# SPACE_PUNCTUATION_REGEXP = re.compile('[' + SPACES + PUNCTUATION + ']+')
DATETIME_REGEXP = re.compile(r'(?:\d+[-/:\s]+)+\d+')
NUMBER_REGEXP = re.compile(r'\d+(?:\.\d+)?')
ZERO_REGEXP = re.compile(r'\b0\b')
WORD_TOKENIZATION_REGEXP = re.compile(
    '((?:[' + ''.join(ARABIC_LETTERS) + ']['+''.join(ARABIC_DIACRITICS)+r']*)+|\d+(?:\.\d+)?)')
SENTENCE_TOKENIZATION_REGEXP = re.compile(r'([' + SENTENCE_SEPARATORS + r'])(?!\w)|' + XML_TAG)
CHAR2INDEX = dict((l, n) for n, l in enumerate(sorted(ARABIC_LETTERS)))
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


def extract_diacritics_2(text):
    """
    Return the diacritics from the text while keeping their original positions including the Shadda marks.
    :param text: str, the diacritized text.
    :return: list, the diacritics. Positions with double diacritics have a tuple as elements.
    """
    assert isinstance(text, str)
    diacritics = []
    for i in range(1, len(text)):
        if text[i] in ARABIC_DIACRITICS:
            if text[i-1] == NAME2DIACRITIC['Shadda']:
                diacritics[-1] = (text[i-1], text[i])
            else:
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
    assert set(diacritics).issubset(ARABIC_DIACRITICS.union(['']))
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


def fix_diacritics_errors(diacritized_text):
    """
    Fix and normalize some diacritization errors in the sentences.
    :param diacritized_text: the text containing the arabic letters with diacritics.
    :return: str, the fixed text.
    """
    assert isinstance(diacritized_text, str)
    # Remove the extra Sukun from ال
    diacritized_text = EXTRA_SUKUN_REGEXP.sub('', diacritized_text)
    # Fix misplaced Fathatan
    diacritized_text = diacritized_text.replace('اً', 'ًا')
    # Fix reversed Shadda-Diacritic
    diacritized_text = DIACRITIC_SHADDA_REGEXP.sub(r'\2\1', diacritized_text)
    # Fix ى that should be ي (disabled)
    # diacritized_text = YA_REGEXP.sub('ي', diacritized_text)
    # Remove the duplicated diacritics by leaving the second one only when there are two incompatible diacritics
    fixed_text = diacritized_text[0]
    for x in diacritized_text[1:]:
        if x in ARABIC_DIACRITICS and fixed_text[-1] in ARABIC_DIACRITICS:
            if fixed_text[-1] != NAME2DIACRITIC['Shadda'] or x == NAME2DIACRITIC['Shadda']:
                fixed_text = fixed_text[:-1]
        # Remove the diacritics that are without letters
        elif x in ARABIC_DIACRITICS and fixed_text[-1] not in ARABIC_LETTERS:
            continue
        fixed_text += x
    return fixed_text


def clean_text(text):
    """
    Remove the unwanted characters from the text.
    :param text: str, the unclean text.
    :return: str, the cleaned text.
    """
    assert isinstance(text, str)
    # Clean HTML garbage, tatweel, dates.
    return DATETIME_REGEXP.sub('', text.replace('ـ', '').replace('&quot;', ''))


def tokenize(sentence):
    """
    Tokenize a sentence into a list of words.
    :param sentence: str, the sentence to be tokenized.
    :return: list of str, list containing the words.
    """
    assert isinstance(sentence, str)
    return list(filter(lambda x: x != '' and x.isprintable(), re.split(WORD_TOKENIZATION_REGEXP, sentence)))


def filter_tokenized_sentence(sentence, min_words=2, min_word_diac_rate=0.8, min_word_diac_ratio=0.5):
    """
    Accept or void a sentence, and clean the tokens.
    :param sentence: the sentence to be filtered.
    :param min_words: minimum number of arabic words that must be left in the cleaned sentence in order to be accepted.
    :param min_word_diac_rate: rate of the diacritized words to the number of arabic words in the sentence.
    :param min_word_diac_ratio: ratio of the diacritized letters to the number of letters in the word.
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
            token = token.strip()
            if not token:
                continue
            word_chars = set(token)
            if word_chars & ARABIC_LETTERS != set():
                arabic_word_count += 1
                word_diacs = extract_diacritics_2(token)
                if len([x for x in word_diacs if x]) / len(word_diacs) >= min_word_diac_ratio:
                    diac_word_count += 1
            new_sentence.append(token)
        if arabic_word_count > 0 and arabic_word_count >= min_words:
            if diac_word_count / arabic_word_count >= min_word_diac_rate:
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
            if line == '':
                continue
            fragments = list(filter(lambda x: x != '',
                                    [x.strip(SPACES) for x in re.split(SENTENCE_TOKENIZATION_REGEXP, line)
                                     if x is not None]))
            if len(fragments) > 1:
                for f1, f2 in zip(fragments[:-1], fragments[1:]):
                    if set(f2).issubset(set(SENTENCE_SEPARATORS)):
                        sentences.append(f1+f2)
                    elif set(f1).issubset(set(SENTENCE_SEPARATORS)):
                        continue
                    else:
                        sentences.append(f1)
            else:
                sentences.extend(fragments)
    return sentences


def text_to_indices(text):
    """
    Transform a sentence into an indices vector.
    :param text: str, the sentence to be transformed.
    :return: ndarray, 1D NumPy array representing the new format.
    """
    assert isinstance(text, str) and set(text) & ARABIC_DIACRITICS == set()
    char_vectors = np.empty((len(text),))
    for i in range(len(text)):
        if text[i] in ARABIC_LETTERS:
            char_vectors[i] = CHAR2INDEX[text[i]]
        elif text[i].isnumeric():
            char_vectors[i] = CHAR2INDEX['0']
        elif text[i].isspace():
            char_vectors[i] = CHAR2INDEX[' ']
    return char_vectors


def add_time_steps(one_hot_matrix, time_steps, word_level):
    """
    Transform a 2D one-hot matrix into a 3D one containing time steps.
    :param one_hot_matrix: ndarray, the one-hot matrix
    :param time_steps: int, the number of time steps
    :param word_level: bool, if True then each instance will represent a word.
    :return: ndarray, 3D matrix with time steps as a second dimension.
    """
    assert isinstance(one_hot_matrix, np.ndarray) and len(one_hot_matrix.shape) == 2
    assert isinstance(time_steps, int) and time_steps > 1
    assert isinstance(word_level, bool)
    space_indices = np.concatenate((np.flatnonzero(np.argmax(one_hot_matrix, axis=1) == CHAR2INDEX[' ']),
                                    np.array(one_hot_matrix.shape[0:1])))
    X = np.empty((
        one_hot_matrix.shape[0] if not word_level else space_indices.shape[0], time_steps, one_hot_matrix.shape[1]
    ))
    offset = time_steps - 1 if not word_level else max(time_steps - space_indices[0], 1)
    padded_one_hot = np.concatenate((np.zeros((offset, one_hot_matrix.shape[1])), one_hot_matrix))
    if not word_level:
        for i in range(X.shape[0]):
            X[i] = padded_one_hot[i:i+time_steps]
    else:
        space_indices += offset
        for i in range(X.shape[0]):
            X[i] = padded_one_hot[space_indices[i]-time_steps:space_indices[i]]
    return X


def input_to_sentence(batch, word_level):
    """
    Revert an input batch again to text format.
    :param batch: ndarray, an input batch representing a sentence.
    :param word_level: bool, True if an instance corresponds to a word, False otherwise.
    :return: str, the original text.
    """
    assert isinstance(batch, np.ndarray) and len(batch.shape) == 3
    assert isinstance(word_level, bool)
    if not word_level:
        one_hot = batch[:, -1]
        indices = np.argmax(one_hot, axis=1)
        return ''.join([INDEX2CHAR[i] for i in indices])
    else:
        sentence = ''
        for row in batch:
            word = []
            i = row.shape[0]-1
            while i > -1 and np.any(row[i]) and np.argmax(row[i]) != CHAR2INDEX[' ']:
                word.append(INDEX2CHAR[np.argmax(row[i])])
                i -= 1
            sentence += ''.join(reversed(word)) + ' '
        return sentence[:-1]
