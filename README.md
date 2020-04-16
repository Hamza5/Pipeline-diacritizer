# Multi-components system for automatic Arabic diacritics restoration

## About

This tool is a command-line application written in Python 3 that automatically add diacritics to
raw undiacritized Arabic text. To accomplish this task, it uses several techniques: Deep Learning, rule-based and
statistical corrections. The deep learning part was implemented using Tensorflow. It was released as a support for the
research paper:
["Multi-components system for automatic Arabic diacritization"](https://doi.org/10.1007/978-3-030-45439-5_23)
which was presented in the [ECIR2020 conference](https://ecir2020.org/program/).

## Installation

This tool is available as a Python 3 package `pipeline-diacritizer` installable through `pip`. For installation
instructions check the
[**Download and installation** wiki page](https://github.com/Hamza5/Pipeline-diacritizer/wiki/Download-and-installation).

## Functions

This tool has 4 main functions: preprocessing of the data, training on the processed data, testing, and
restoring the diacritics of an undiacritized text. In addition, it can calculates some statistics on a given dataset and
the ratio of Out-of-Vocabulary words in a testing set according to a train set.

This is a quick introduction to the most important ones, without mentioning all the possible options for each one. For
additional options, consider calling any subcommand with the option `--help` or `-h` (ex:
`pipeline_diacritizer train --help`) or [check the wiki](https://github.com/Hamza5/Pipeline-diacritizer/wiki)
for more details.

### Preprocessing

Before feeding the new data to this application for training or testing, it needs to be converted to the standard format
of this application: one sentence per line, where a sentence is delimited by a dot, a comma, or an end of line
character.

```
$ pipeline_diacritizer preprocess <source_file> <destination_file>
```

If the data is not yet partitioned into training, validation and testing sets, the program can help in this task using
the following command:

```
$ pipeline_diacritizer partition <dataset_file>
```

### Training

To run the training and validation on selected training/validation sets, use the next command:

```
$ pipeline_diacritizer train --train-data <train_file> --val-data <val_file>
```

### Testing

To evaluate the performances of the application on a testing set, use this command:

```
$ pipeline_diacritizer test <test_file>
```

### Diacritization

The following command restores the diacritics of the Arabic words from the supplied text file and outputs a diacritized
copy:

```
$ pipeline_diacritizer diacritize <text_file>
```

### Statistics

To get some statistics about the dataset, such as the count of tokens, arabic words, numbers... use the following
command:

```
$ pipeline_diacritizer stat <dataset_file>
```

### OoV Counting

To calculate the ratio of the Out-of-Vocabulary words between the train set and the validation/test set, use the next
command:

```
$ pipeline_diacritizer oov <train_file> <test_file>
```

## License

Pipeline-diacritizer code is licensed under
[MIT License](https://github.com/Hamza5/Pipeline-diacritizer/blob/master/LICENSE.txt).
