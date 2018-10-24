"""
Module containing several functions to read and correct several errors in Tashkeela dataset and converting its data.
"""

# Hexadecimal values taken from https://www.unicode.org/charts/PDF/U0600.pdf
D_NAMES = ['Fathatan', 'Dammatan', 'Kasratan', 'Fatha', 'Damma', 'Kasra', 'Shadda', 'Sukun']
NAME2DIACRITIC = dict((name, chr(code)) for name, code in zip(D_NAMES, range(0x064B, 0x0653)))
DIACRITIC2NAME = dict((code, name) for name, code in NAME2DIACRITIC.items())
ARABIC_DIACRITICS = frozenset(NAME2DIACRITIC.values())
ARABIC_LETTERS = frozenset(chr(x) for x in (list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B))))
ARABIC_SYMBOLS = ARABIC_LETTERS | ARABIC_DIACRITICS


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
