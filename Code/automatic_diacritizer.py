from random import shuffle

from diacritization_model import DiacritizationModel


def process(source, destination, min_words, min_diac_rate, max_chars_count):
    with destination.open('w', encoding='UTF-8') as dest_file:
        if source.is_dir():
            for file_path in filter(lambda x: x.is_file(), source.iterdir()):
                print('Parsing', file_path, '...')
                sentences = read_text_file(str(file_path))
                filtered_sentences = set()
                for sf in filter(lambda x: len(x) > 0,
                                 [filter_tokenized_sentence(tokenize(fix_diacritics_errors(s)),
                                                            min_words, min_diac_rate)
                                  for s in sentences]):
                    filtered_sentences.add(' '.join(sf))
                for sf in filtered_sentences:
                    print(sf[:max_chars_count].rstrip(), file=dest_file)
        elif source.is_file():
            print('Parsing', source, '...')
            sentences = read_text_file(str(source))
            filtered_sentences = set()
            for sf in filter(lambda x: len(x) > 0,
                             [filter_tokenized_sentence(tokenize(fix_diacritics_errors(s)),
                                                        min_words, min_diac_rate)
                              for s in sentences]):
                filtered_sentences.add(' '.join(sf))
            for sf in filtered_sentences:
                print(sf[:max_chars_count].rstrip(), file=dest_file)
        else:
            root_p.error('{} is neither a file nor a directory!'.format(source))
            root_p.exit(-2)
    print('Finished')


def partition(dataset_file, train_ratio, val_test_ratio, shuffle_every):
    # Prepare files for train, validation and test
    train_path = dataset_file.with_name(dataset_file.stem + '_train.txt')
    val_path = dataset_file.with_name(dataset_file.stem + '_val.txt')
    test_path = dataset_file.with_name(dataset_file.stem + '_test.txt')
    train_path.open('w').close()
    val_path.open('w').close()
    test_path.open('w').close()
    print('Generating sets from', dataset_file)
    with dataset_file.open('r', encoding='UTF-8') as data_file:
        sentences = []
        for line in data_file:
            sentences.append(line)
            if len(sentences) % shuffle_every == 0:
                train_size = round(train_ratio * len(sentences))
                val_size = round(val_test_ratio * (len(sentences) - train_size))
                shuffle(sentences)
                with train_path.open('a', encoding='UTF-8') as train_file:
                    for s in sentences[:train_size]:
                        train_file.write(s)
                with val_path.open('a', encoding='UTF-8') as val_file:
                    for s in sentences[train_size:train_size + val_size]:
                        val_file.write(s)
                with test_path.open('a', encoding='UTF-8') as test_file:
                    for s in sentences[train_size + val_size:]:
                        test_file.write(s)
                sentences.clear()
        train_size = round(train_ratio * len(sentences))
        val_size = round(val_test_ratio * (len(sentences) - train_size))
        shuffle(sentences)
        with train_path.open('a', encoding='UTF-8') as train_file:
            for s in sentences[:train_size]:
                train_file.write(s)
        with val_path.open('a', encoding='UTF-8') as val_file:
            for s in sentences[train_size:train_size + val_size]:
                val_file.write(s)
        with test_path.open('a', encoding='UTF-8') as test_file:
            for s in sentences[train_size + val_size:]:
                test_file.write(s)
    print('Finished')


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


def test(test_data_path, weights_dir):
    test_data = []
    with test_data_path.open('r', encoding='UTF-8') as test_data_file:
        for line in test_data_file:
            test_data.append(line.rstrip('\n'))
    model = DiacritizationModel(str(weights_dir))
    model.load()
    model.test(test_data)


if __name__ == '__main__':

    import sys
    from pathlib import Path
    from argparse import ArgumentParser

    from dataset_preprocessing import read_text_file, filter_tokenized_sentence, tokenize, fix_diacritics_errors

    root_p = ArgumentParser(description='Script to generate the datasets and train the diacritization models.')
    subparsers = root_p.add_subparsers(title='Commands', description='Available operations')
    preprocessing_p = subparsers.add_parser('process',
                                            description='Transform Arabic raw text files to a preprocessed dataset by'
                                                        ' splitting sentences, dropping punctuation and noise, and '
                                                        'normalizing the spaces and the numbers, then keeping only the '
                                                        'highly diacritized sentences.')
    preprocessing_p.add_argument('source', type=Path, help='Path of a raw text file or a folder containing the text '
                                                           'files.')
    preprocessing_p.add_argument('destination', type=Path, help='Path of the generated text file after processing.')
    preprocessing_p.add_argument('--min-words', '-w', type=int, default=2,
                                 help='Minimum number of arabic words that must be left in the cleaned sentence in '
                                      'order to be accepted.')
    preprocessing_p.add_argument('--min-diac-rate', '-d', type=float, default=1,
                                 help='Minimum rate of the diacritized words to the number of arabic words in the '
                                      'sentence.')
    preprocessing_p.add_argument('--max-chars-count', '-c', type=int, default=2000,
                                 help='Maximum number of characters to keep in a long sentence.')
    partition_p = subparsers.add_parser('partition', help='Divide a dataset to train, validation and test fragments.')
    partition_p.add_argument('dataset_file', type=Path, help='The preprocessed dataset file.')
    partition_p.add_argument('--train-ratio', '-t', type=float, default=0.9, help='Ratio of data for training.')
    partition_p.add_argument('--val-test-ratio', '-v', type=float, default=0.5, help='Split ratio between validation '
                                                                                     'and test data.')
    partition_p.add_argument('--shuffle-every', '-s', type=int, default=1000,
                             help='Number of sentences to accumulate before shuffling.')
    train_parser = subparsers.add_parser('train', description='Launch the training of a model.')
    train_parser.add_argument('--train-data', '-t', type=Path, required=True, help='Training dataset.')
    train_parser.add_argument('--val-data', '-v', type=Path, required=True, help='Validation dataset.')
    train_parser.add_argument('--iterations', '-i', type=int, default=15,
                              help='Maximum number of iterations for training.')
    train_parser.add_argument('--weights-dir', '-w', type=Path, default=Path.cwd(),
                              help='Directory containing the weights file for the model.')
    train_parser.add_argument('--early-stop', '-e', type=int, default=3,
                              help='Maximum number of tries to add when the model performances does not improve.')
    test_parser = subparsers.add_parser('test', description='Test a pretrained model.')
    test_parser.add_argument('--test-data', '-s', type=Path, required=True, help='Test dataset.')
    test_parser.add_argument('--weights-dir', '-w', type=Path, default=Path.cwd(),
                             help='Directory containing the weights file for the model.')
    args = root_p.parse_args()
    if not vars(args):
        root_p.print_help(sys.stderr)
        root_p.exit(-1)
    if 'source' in vars(args):
        process(args.source, args.destination, args.min_words, args.min_diac_rate, args.max_chars_count)
    elif 'dataset_file' in vars(args):
        partition(args.dataset_file, args.train_ratio, args.val_test_ratio, args.shuffle_every)
    elif 'train_data' in vars(args):
        train(args.train_data, args.val_data, args.iterations, args.weights_dir, args.early_stop)
    elif 'test_data' in vars(args):
        test(args.test_data, args.weights_dir)
