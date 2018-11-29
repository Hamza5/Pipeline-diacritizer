if __name__ == '__main__':

    import sys
    from pathlib import Path
    from random import shuffle
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
    preprocessing_p.add_argument('--min-diac-rate', '-d', type=float, default=0.8,
                                 help='Minimum rate of the diacritized words to the number of arabic words in the '
                                      'sentence.')
    partition_p = subparsers.add_parser('partition', help='Divide a dataset to train, validation and test fragments.')
    partition_p.add_argument('dataset_file', type=Path, help='The preprocessed dataset file.')
    partition_p.add_argument('--train-ratio', '-t', type=float, default=0.9, help='Ratio of data for training.')
    partition_p.add_argument('--val-test-ratio', '-v', type=float, default=0.5, help='Split ratio between validation '
                                                                                     'and test data.')
    partition_p.add_argument('--shuffle-every', '-s', type=int, default=1000,
                             help='Number of sentences to accumulate before shuffling.')
    args = root_p.parse_args()
    if not vars(args):
        root_p.print_help(sys.stderr)
        root_p.exit(-1)
    if 'source' in vars(args):
        with args.destination.open('w', encoding='UTF-8') as dest_file:
            if args.source.is_dir():
                for file_path in filter(lambda x: x.is_file(), args.source.iterdir()):
                    print('Parsing', file_path, '...')
                    sentences = read_text_file(str(file_path))
                    filtered_sentences = set()
                    for sf in filter(lambda x: len(x) > 0,
                                     [filter_tokenized_sentence(tokenize(fix_diacritics_errors(s)),
                                                                args.min_words, args.min_diac_rate)
                                      for s in sentences]):
                        filtered_sentences.add(' '.join(sf))
                    for sf in filtered_sentences:
                        print(sf, file=dest_file)
            elif args.source.is_file():
                print('Parsing', args.source, '...')
                sentences = read_text_file(str(args.source))
                filtered_sentences = set()
                for sf in filter(lambda x: len(x) > 0,
                                 [filter_tokenized_sentence(tokenize(fix_diacritics_errors(s)),
                                                            args.min_words, args.min_diac_rate)
                                  for s in sentences]):
                    filtered_sentences.add(' '.join(sf))
                for sf in filtered_sentences:
                    print(sf, file=dest_file)
            else:
                root_p.error('{} is neither a file nor a directory!'.format(args.source))
                root_p.exit(-2)
        print('Finished')
    elif 'dataset_file' in vars(args):
        # Prepare files for train, validation and test
        train_path = args.dataset_file.with_name(args.dataset_file.stem + '_train.txt')
        val_path = args.dataset_file.with_name(args.dataset_file.stem + '_val.txt')
        test_path = args.dataset_file.with_name(args.dataset_file.stem + '_test.txt')
        train_path.open('w').close()
        val_path.open('w').close()
        test_path.open('w').close()
        print('Generating sets from', args.dataset_file)
        with args.dataset_file.open('r', encoding='UTF-8') as data_file:
            sentences = []
            for line in data_file:
                sentences.append(line)
                if len(sentences) % args.shuffle_every == 0:
                    train_size = round(args.train_ratio * len(sentences))
                    val_size = round(args.val_test_ratio * (len(sentences) - train_size))
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
            train_size = round(args.train_ratio * len(sentences))
            val_size = round(args.val_test_ratio * (len(sentences) - train_size))
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
