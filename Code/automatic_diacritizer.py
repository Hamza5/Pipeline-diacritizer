if __name__ == '__main__':

    import sys
    from pathlib import Path
    from argparse import ArgumentParser

    from dataset_preprocessing import read_text_file, filter_tokenized_sentence, tokenize, fix_double_diacritics_error

    p = ArgumentParser(description='Script to generate the datasets and train the diacritization models.')
    subparsers = p.add_subparsers(title='Commands', description='Available operations')
    generation_parser = subparsers.add_parser('generate-dataset')
    generation_parser.add_argument('source-directory', type=Path, help='Path of the folder containing raw text files.')
    generation_parser.add_argument('destination-file', type=Path, help='Path of the output file.')
    args = vars(p.parse_args())
    if not args:
        p.print_help(sys.stderr)
        p.exit(-1)
    if 'source-directory' in args:
        with args['destination-file'].open('w', encoding='UTF-8') as dest_file:
            for file_path in filter(lambda x: x.is_file(), args['source-directory'].iterdir()):
                print('Parsing', file_path, '...')
                sentences = read_text_file(str(file_path))
                for sf in filter(lambda x: len(x) > 0,
                                 [filter_tokenized_sentence(tokenize(fix_double_diacritics_error(s)))
                                  for s in sentences]):
                    dest_file.write(' '.join(sf))
        print('Finished')
