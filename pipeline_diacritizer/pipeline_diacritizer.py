#!/usr/bin/python3
import os
import sys
import random
from itertools import chain
from pathlib import Path
from argparse import ArgumentParser, FileType
from pkg_resources import resource_filename

from pipeline_diacritizer.dataset_preprocessing import WORD_TOKENIZATION_REGEXP, NUMBER_REGEXP, extract_diacritics,\
    clear_diacritics, ARABIC_DIACRITICS, ARABIC_LETTERS, SENTENCE_TOKENIZATION_REGEXP, SPACES, ARABIC_SYMBOLS,\
    read_text_file, filter_tokenized_sentence, tokenize, fix_diacritics_errors
from pipeline_diacritizer.diacritization_model import DiacritizationModel

PARAMS_DIR = 'Tashkeela_params'


def print_progress_bar(current, maximum):
    assert isinstance(current, int)
    assert isinstance(maximum, int)
    progress_text = '[{:50s}] {:d}/{:d} ({:0.2%})'.format('=' * int(current / maximum * 50), current, maximum,
                                                          current / maximum)
    sys.stdout.write('\r' + progress_text)
    sys.stdout.flush()


def preprocess(source, destination, min_words, ratio_diac_words, max_chars_count, ratio_diac_letters):
    with destination.open('w', encoding='UTF-8') as dest_file:
        if source.is_dir():
            sentences = []
            for (dirpath, dirnames, filenames) in os.walk(source):
                for file_name in filenames:
                    file_path = os.path.join(dirpath, file_name)
                    print('Parsing {} ...'.format(file_path))
                    sentences.extend(read_text_file(str(file_path)))
        elif source.is_file():
            print('Parsing {} ...'.format(source))
            sentences = read_text_file(str(source))
        else:
            print('{} is neither a file nor a directory!'.format(source), file=sys.stderr)
            sys.exit(-2)
        print('Preprocessing sentences...')
        filtered_sentences = set()
        for i, sf in enumerate(map(lambda s: filter_tokenized_sentence(tokenize(fix_diacritics_errors(s)), min_words,
                                                                       ratio_diac_words, ratio_diac_letters),
                                   sentences), 1):
            if len(sf) > 0:
                filtered_sentences.add(' '.join(sf))
            print_progress_bar(i, len(sentences))
        print(file=sys.stderr)
        print('Generating file {} ...'.format(destination))
        for sf in filtered_sentences:
            print(sf[:max_chars_count].rstrip(), file=dest_file)
    print('Pre-processing finished successfully.')


def partition(dataset_file, train_ratio, val_test_ratio, shuffle_every):

    def write_train_val_test_parts(train_path, val_path, test_path, sentences):
        train_size = round(train_ratio * len(sentences))
        val_size = round(val_test_ratio * (len(sentences) - train_size))
        random.shuffle(sentences)
        with train_path.open('a', encoding='UTF-8') as train_file:
            for s in sentences[:train_size]:
                train_file.write(s)
        with val_path.open('a', encoding='UTF-8') as val_file:
            for s in sentences[train_size:train_size + val_size]:
                val_file.write(s)
        with test_path.open('a', encoding='UTF-8') as test_file:
            for s in sentences[train_size + val_size:]:
                test_file.write(s)

    # Prepare files for train, validation and test
    train_path = dataset_file.with_name(dataset_file.stem + '_train.txt')
    val_path = dataset_file.with_name(dataset_file.stem + '_val.txt')
    test_path = dataset_file.with_name(dataset_file.stem + '_test.txt')
    train_path.open('w').close()
    val_path.open('w').close()
    test_path.open('w').close()
    print('Generating sets from {} ...'.format(dataset_file))
    with dataset_file.open('r', encoding='UTF-8') as data_file:
        sentences = []
        for line in data_file:
            sentences.append(line)
            if len(sentences) % shuffle_every == 0:
                write_train_val_test_parts(train_path, val_path, test_path, sentences)
                print('{} sentences written.'.format(len(sentences)))
                sentences.clear()
        write_train_val_test_parts(train_path, val_path, test_path, sentences)
        print('{} sentences written'.format(len(sentences)))
    print('Partitioning finished successfully.')


def train(train_data_path, val_data_path, iterations, weights_dir, early_stop):
    train_data = []
    val_data = []
    with train_data_path.open('r', encoding='UTF-8') as train_data_file:
        for line in train_data_file:
            train_data.append(line.rstrip('\n'))
    with val_data_path.open('r', encoding='UTF-8') as val_data_file:
        for line in val_data_file:
            val_data.append(line.rstrip('\n'))
    model = DiacritizationModel(str(weights_dir))
    model.load()
    model.train(train_data, val_data, iterations, early_stop)


def test(test_data_path, weights_dir, arabic_only, include_no_diacritic, enable_rules, enable_trigrams, enable_bigrams,
         enable_unigrams, enable_patterns):
    test_data = []
    print('Loading test dataset...')
    with test_data_path.open('r', encoding='UTF-8') as test_data_file:
        for line in test_data_file:
            test_data.append(line.rstrip('\n'))
    model = DiacritizationModel(str(weights_dir), enable_rules, enable_trigrams, enable_bigrams, enable_unigrams,
                                enable_patterns)
    model.load()
    print('Testing...')
    model.test(test_data, arabic_only, include_no_diacritic)


def diacritize(text_path, weights_dir, output_file, enable_rules, enable_trigrams, enable_bigrams, enable_unigrams,
               enable_patterns):
    assert isinstance(text_path, Path)
    assert isinstance(weights_dir, Path)
    model = DiacritizationModel(str(weights_dir), enable_rules, enable_trigrams, enable_bigrams, enable_unigrams,
                                enable_patterns)
    model.load()
    with text_path.open('rt', encoding='UTF-8') as text_file:
        for line in text_file:
            sentences = list(filter(lambda x: x != '',
                                    [x.strip(SPACES) for x in SENTENCE_TOKENIZATION_REGEXP.split(line)
                                     if x is not None]))
            d_sentences = []
            for sentence in sentences:
                if set(sentence) & ARABIC_LETTERS != set():
                    d_sentences.append(model.diacritize_original(sentence))
                else:
                    d_sentences.append(sentence)
            for s, d_s in zip(sentences, d_sentences):
                line = line.replace(s, d_s)
            print(line, file=output_file, end='')


def stat(file_path):
    assert isinstance(file_path, Path)
    chars_count = 0
    arabic_letters_count = 0
    digits_count = 0
    tokens_count = 0
    numbers_count = 0
    arabic_words_count = 0
    diacritics_count = 0
    diacritization_forms = {}
    with file_path.open('r', encoding='UTF-8') as data_file:
        for line in data_file:
            line = line.rstrip('\n')
            segments = [x.strip() for x in WORD_TOKENIZATION_REGEXP.split(line) if x.strip() != '']
            for seg in segments:
                tokens_count += 1
                chars_count += len(seg)
                if WORD_TOKENIZATION_REGEXP.match(seg):
                    if NUMBER_REGEXP.match(seg):
                        numbers_count += 1
                        digits_count += sum(1 for x in seg if x in '0123456789')
                    else:
                        arabic_words_count += 1
                        undiacritized = clear_diacritics(seg)
                        arabic_letters_count += len(undiacritized)
                        if undiacritized != seg:
                            try:
                                diacritization_forms[undiacritized].add(seg)
                            except KeyError:
                                diacritization_forms[undiacritized] = {seg}
                            diacritics_count += len([x for x in extract_diacritics(seg) if x in ARABIC_DIACRITICS])
    print('Statistics about the dataset:', file_path)
    print('-'*35)
    print('|Characters         |{:13d}|'.format(chars_count))
    print('|Tokens             |{:13d}|'.format(tokens_count))
    print('|Numbers            |{:13d}|'.format(numbers_count))
    print('|Digits             |{:13d}|'.format(digits_count))
    print('|Arabic words       |{:13d}|'.format(arabic_words_count))
    print('|Arabic letters     |{:13d}|'.format(arabic_letters_count))
    print('|Diacritics         |{:13d}|'.format(diacritics_count))
    print('|Undiacritized forms|{:13d}|'.format(len(diacritization_forms)))
    print('|Diacritized forms  |{:13d}|'.format(sum(len(x) for x in diacritization_forms.values())))
    print('-'*35)


def oov_count(train_set_path, test_set_path):
    assert isinstance(train_set_path, Path)
    assert isinstance(test_set_path, Path)
    with train_set_path.open(encoding='UTF-8') as train_set_file:
        train_set_sentences = list(map(str.split, train_set_file.readlines()))
    train_set_d_words = set(filter(lambda w: set(w).issubset(ARABIC_SYMBOLS), chain(*train_set_sentences)))
    with test_set_path.open(encoding='UTF-8') as test_set_file:
        test_set_sentences = list(map(str.split, test_set_file.readlines()))
    test_set_d_words = set(filter(lambda w: set(w).issubset(ARABIC_SYMBOLS), chain(*test_set_sentences)))
    diacritized_words_oov = test_set_d_words - train_set_d_words
    train_set_u_words = set(map(clear_diacritics, train_set_d_words))
    test_set_u_words = set(map(clear_diacritics, test_set_d_words))
    u_words_oov = test_set_u_words - train_set_u_words
    print('Diacritized OoV Arabic words: {}/{} ({:.2%})'.format(len(diacritized_words_oov), len(test_set_d_words),
                                                                len(diacritized_words_oov) / len(test_set_d_words)))
    print('Undiacritized OoV Arabic words: {}/{} ({:.2%})'.format(len(u_words_oov), len(test_set_u_words),
                                                                  len(u_words_oov) / len(test_set_u_words)))


def main():
    root_p = ArgumentParser(description='Program allowing to process diacritized datasets, training, testing and '
                                        'diacritizing.')
    subparsers = root_p.add_subparsers(title='Commands', description='Available operations')
    preprocessing_p = subparsers.add_parser('preprocess',
                                            help='Transform Arabic raw text files to a preprocessed dataset by '
                                                 'splitting sentences, dropping punctuation and noise, normalizing '
                                                 'spaces and numbers, then keeping only the highly diacritized '
                                                 'sentences.')
    preprocessing_p.add_argument('source', type=Path,
                                 help='Path of a raw text file or a folder containing the text files.')
    preprocessing_p.add_argument('destination', type=Path, help='Path of the generated text file after processing.')
    preprocessing_p.add_argument('--min-words', '-w', type=int, default=2,
                                 help='Minimum number of arabic words that must be left in the cleaned sentence in '
                                      'order to be accepted.')
    preprocessing_p.add_argument('--min-diac-words-ratio', '-d', type=float, default=1,
                                 help='Minimum rate of the diacritized words to the number of arabic words in the '
                                      'sentence.')
    preprocessing_p.add_argument('--min-diac-letters-ratio', '-l', type=float, default=0.5,
                                 help='Minimum ratio of the diacritized letters to the number of the letters in the '
                                      'word.')
    preprocessing_p.add_argument('--max-chars-count', '-c', type=int, default=2000,
                                 help='Maximum number of characters to keep in a long sentence.')
    partition_p = subparsers.add_parser('partition', help='Divide a dataset to train, validation and test fragments.')
    partition_p.add_argument('dataset_file', type=Path, help='The preprocessed dataset file.')
    partition_p.add_argument('--train-ratio', '-t', type=float, default=0.9, help='Ratio of data for training.')
    partition_p.add_argument('--val-test-ratio', '-v', type=float, default=0.5,
                             help='Split ratio between validation and test data.')
    partition_p.add_argument('--shuffle-every', '-s', type=int, default=100000,
                             help='Number of sentences to accumulate before shuffling.')
    train_parser = subparsers.add_parser('train', help='Launch the training of a model.')
    train_parser.add_argument('--train-data', '-t', type=Path, required=True, help='Training dataset.')
    train_parser.add_argument('--val-data', '-v', type=Path, required=True, help='Validation dataset.')
    train_parser.add_argument('--iterations', '-i', type=int, default=15,
                              help='Maximum number of iterations for training.')
    train_parser.add_argument('--weights-dir', '-w', type=Path,
                              default=resource_filename(__package__, PARAMS_DIR),
                              help='Directory containing the weights file for the model.')
    train_parser.add_argument('--early-stop', '-e', type=int, default=3,
                              help='Maximum number of tries to add when the model performances does not improve.')
    test_diacritize_parent = ArgumentParser(add_help=False)
    test_diacritize_parent.add_argument('--disable-rules', dest='rules', action='store_false',
                                        help='Do not use the rules.')
    test_diacritize_parent.add_argument('--disable-trigrams', dest='trigrams', action='store_false',
                                        help='Do not load the trigrams.')
    test_diacritize_parent.add_argument('--disable-bigrams', dest='bigrams', action='store_false',
                                        help='Do not load the bigrams.')
    test_diacritize_parent.add_argument('--disable-unigrams', dest='unigrams', action='store_false',
                                        help='Do not load the unigrams.')
    test_diacritize_parent.add_argument('--disable-patterns', dest='patterns', action='store_false',
                                        help='Do not load the patterns.')
    test_diacritize_parent.add_argument('--weights-dir', '-w', type=Path,
                                        default=resource_filename(__package__, PARAMS_DIR),
                                        help='Directory containing the weights file for the model.')
    test_parser = subparsers.add_parser('test', help='Test a pretrained model.', parents=[test_diacritize_parent])
    test_parser.add_argument('test_data', type=Path, help='Test dataset.')
    test_parser.add_argument('--all-characters', '-a', action='store_false', dest='arabic_only',
                             help='Include the non-Arabic symbols in the calculation of the metrics')
    test_parser.add_argument('--ignore-no-diacritics', '-n', action='store_false', dest='no_diacritic',
                             help='Include the non-Arabic symbols in the calculation of the metrics')
    diacritize_parser = subparsers.add_parser('diacritize', parents=[test_diacritize_parent],
                                              help='Restore the diacritics of the Arabic letters in a text.')
    diacritize_parser.add_argument('text_file', type=Path, help='A text file with an undiacritized Arabic text.')
    diacritize_parser.add_argument('--output-file', '-o', type=FileType('wt', encoding='UTF-8'), default=sys.stdout,
                                   help='The output file for the results.')
    stat_parser = subparsers.add_parser('stat', help='Calculate some statistics about a dataset.')
    stat_parser.add_argument('dataset_text_file', type=Path, help='The file path of the dataset.')
    oov_parser = subparsers.add_parser('oov', help='Calculate the ratio of OoV words in a test set.')
    oov_parser.add_argument('train_set', type=Path, help='The file path of the training set.')
    oov_parser.add_argument('test_set', type=Path, help='The file path of the test/val set.')
    args = root_p.parse_args()
    if not vars(args):
        root_p.print_help(sys.stderr)
        root_p.exit(-1)
    if 'source' in vars(args):
        preprocess(args.source, args.destination, args.min_words, args.min_diac_words_ratio, args.max_chars_count,
                   args.min_diac_letters_ratio)
    elif 'dataset_file' in vars(args):
        partition(args.dataset_file, args.train_ratio, args.val_test_ratio, args.shuffle_every)
    elif 'train_data' in vars(args):
        train(args.train_data, args.val_data, args.iterations, args.weights_dir, args.early_stop)
    elif 'test_data' in vars(args):
        test(args.test_data, args.weights_dir, args.arabic_only, args.no_diacritic, args.rules, args.trigrams,
             args.bigrams, args.unigrams, args.patterns)
    elif 'text_file' in vars(args):
        diacritize(args.text_file, args.weights_dir, args.output_file, args.rules, args.trigrams, args.bigrams,
                   args.unigrams, args.patterns)
    elif 'dataset_text_file' in vars(args):
        stat(args.dataset_text_file)
    elif 'train_set' in vars(args):
        oov_count(args.train_set, args.test_set)
