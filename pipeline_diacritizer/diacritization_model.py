"""
Module containing the new diacritization model.
"""
import os.path
import pickle
import re
import sys
from collections import Iterable
from datetime import datetime

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, EarlyStopping, TerminateOnNaN
from tensorflow.keras.layers import LSTM, Dense, Flatten, Bidirectional, Input, Layer, Lambda
from tensorflow.keras.metrics import binary_accuracy, categorical_accuracy
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.utils import Sequence, to_categorical

from pipeline_diacritizer.dataset_preprocessing import NAME2DIACRITIC, CHAR2INDEX, extract_diacritics_2, \
    clear_diacritics, add_time_steps, NUMBER_REGEXP, WORD_TOKENIZATION_REGEXP, ZERO_REGEXP, ARABIC_LETTERS


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision. Computes the precision, a
    metric for multi-label classification of how many selected items are
    relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall. Computes the recall, a metric
    for multi-label classification of how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


class DiacritizationModel:
    """
    Class containing the required functions for training, testing and making predictions for an automatic diacritization
    system.
    """

    TIME_STEPS = 10
    OPTIMIZER = Adadelta()

    def __init__(self, save_dir='.', use_rules=True, use_trigrams=True, use_bigrams=True, use_unigrams=True,
                 use_patterns=True):
        """
        Construct a automatic diacritization system model.
        :param save_dir: the path of the directory containing the weights and the history files.
        """
        self.save_dir = save_dir
        self.rules_enabled = use_rules
        self.trigrams_enabled = use_trigrams
        self.bigrams_enabled = use_bigrams
        self.unigrams_enabled = use_unigrams
        self.patterns_enabled = use_patterns
        self.input_layer = Input(shape=(self.TIME_STEPS, len(CHAR2INDEX)))
        self.inner_layers = [
            Bidirectional(LSTM(64, return_sequences=True, unroll=True, dropout=0.1), name='L1'),
            Bidirectional(LSTM(64, return_sequences=True, unroll=True, dropout=0.1), name='L2'),
            Flatten(name='F'),
            (Dense(8, activation='tanh', name='D1'), Dense(64, activation='tanh', name='D2'))
        ]
        previous_layer = self.input_layer
        for layer in self.inner_layers[:-1]:
            previous_layer = layer(previous_layer)
        shadda_side, haraka_side = self.inner_layers[-1]
        shadda_side = shadda_side(previous_layer)
        haraka_side = haraka_side(previous_layer)
        self.output_shadda_layer = Dense(1, activation='sigmoid', name='D3')(shadda_side)
        self.output_haraka_layer = Dense(8, activation='softmax', name='D4')(haraka_side)
        self.shadda_corrections_layer = Lambda(self.shadda_post_corrections, name='output_shadda')(
            [self.input_layer, self.output_shadda_layer, self.output_haraka_layer]
        )
        self.haraka_corrections_layer = Lambda(self.haraka_post_corrections, name='output_haraka')(
            [self.input_layer, self.output_haraka_layer, self.output_shadda_layer]
        )
        self.model = Model(inputs=self.input_layer,
                           outputs=[self.shadda_corrections_layer, self.haraka_corrections_layer])
        self.model.compile(optimizer=self.OPTIMIZER,
                           loss={'output_haraka': 'categorical_crossentropy', 'output_shadda': 'binary_crossentropy'},
                           metrics={'output_haraka': [categorical_accuracy, precision, recall],
                                    'output_shadda': [binary_accuracy, precision, recall]})
        self.values_history = dict((k, []) for k in self.model.metrics_names + ['val_'+x for x in
                                                                                self.model.metrics_names])
        if os.path.isfile(self.get_history_file_path()):
            with open(self.get_history_file_path(), 'rb') as history_file:
                self.values_history = pickle.load(history_file)
        self.trigram_context = {}
        self.bigram_context = {}
        self.undiacritized_vocabulary = {}
        self.patterns = {}

    def get_weights_file_path(self):
        layer_names_shapes = [l.name + '#' + str(l.output_shape[-1])
                              if isinstance(l, Layer)
                              else ','.join([sl.name + '#' + str(sl.output_shape[-1]) for sl in l])
                              for l in self.inner_layers]
        return os.path.join(self.save_dir, type(self).__name__ + '_' + '_'.join(layer_names_shapes) + '.h5')

    def get_history_file_path(self):
        return self.get_weights_file_path()[:-3]+'_history.pkl'

    def get_trigrams_file_path(self):
        return os.path.join(self.save_dir, type(self).__name__ + '_trigrams.pkl')

    def get_bigrams_file_path(self):
        return os.path.join(self.save_dir, type(self).__name__ + '_bigrams.pkl')

    def get_unigrams_file_path(self):
        return os.path.join(self.save_dir, type(self).__name__ + '_unigrams.pkl')

    def get_patterns_file_path(self):
        return os.path.join(self.save_dir, type(self).__name__ + '_patterns.pkl')

    @staticmethod
    def levenshtein_distance(s1, s2):
        # From https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance
        if len(s1) < len(s2):
            return __class__.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[
                                 j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def diacritic_to_index(diacritic):
        return ['', NAME2DIACRITIC['Fatha'], NAME2DIACRITIC['Damma'],  NAME2DIACRITIC['Kasra'], NAME2DIACRITIC['Sukun'],
                NAME2DIACRITIC['Fathatan'], NAME2DIACRITIC['Dammatan'], NAME2DIACRITIC['Kasratan']].index(diacritic)

    @staticmethod
    def index_to_diacritic(index):
        return ['', NAME2DIACRITIC['Fatha'], NAME2DIACRITIC['Damma'],  NAME2DIACRITIC['Kasra'], NAME2DIACRITIC['Sukun'],
                NAME2DIACRITIC['Fathatan'], NAME2DIACRITIC['Dammatan'], NAME2DIACRITIC['Kasratan']][index]

    @staticmethod
    def generate_dataset(sentences):
        assert isinstance(sentences, Iterable)
        targets = []
        inputs = []
        for sentence in sentences:
            diacritics = extract_diacritics_2(sentence)
            letters_text = clear_diacritics(sentence)
            shadda_positions = []
            harakat_indices = []
            for d in diacritics:
                shadda_positions.append(1 if d and d[0] == NAME2DIACRITIC['Shadda'] else 0)
                harakat_indices.append(DiacritizationModel.diacritic_to_index(d[-1])
                                       if d and d[-1] != NAME2DIACRITIC['Shadda'] else 0)
            text_indices = [CHAR2INDEX[x] for x in letters_text]
            targets.append((shadda_positions, harakat_indices))
            inputs.append(text_indices)
        return inputs, targets

    def shadda_post_corrections(self, in_out):
        """
        Drop any obviously misplaced shadda marks according to the character and its context.
        :param in_out: input layer and prediction layers outputs.
        :return: corrected predictions.
        """
        inputs, pred_shadda, pred_haraka = in_out
        if not self.rules_enabled:
            return pred_shadda
        # Drop the shadda from the forbidden letters
        forbidden_chars = [CHAR2INDEX[' '], CHAR2INDEX['ا'], CHAR2INDEX['ء'], CHAR2INDEX['أ'], CHAR2INDEX['إ'],
                           CHAR2INDEX['آ'], CHAR2INDEX['ى'], CHAR2INDEX['ئ'], CHAR2INDEX['ة'], CHAR2INDEX['0']]
        char_index = K.argmax(inputs[:, -1], axis=-1)
        allowed_instances = K.cast(K.not_equal(char_index, forbidden_chars[0]), 'float32')
        for char in forbidden_chars[1:]:
            allowed_instances *= K.cast(K.not_equal(char_index, char), 'float32')
        allowed_instances *= K.sum(inputs[:, -2], axis=-1)  # Special requirement for the first letter in the sentence.
        # Drop the shadda from the letter following the space
        previous_char_index = K.argmax(inputs[:, -2], axis=-1)
        allowed_instances *= K.cast(K.not_equal(previous_char_index, CHAR2INDEX[' ']), 'float32')
        # Drop the shadda from the letter having a Sukun
        allowed_instances *= K.cast(K.not_equal(K.argmax(pred_haraka, axis=1), 4), 'float32')
        return K.reshape(allowed_instances, (-1, 1)) * pred_shadda

    def haraka_post_corrections(self, in_out):
        """
        Change any obviously wrong haraka marks according to the character and its context.
        :param in_out: input layer and prediction layers outputs.
        :return: corrected predictions.
        """
        inputs, pred_haraka, pred_shadda = in_out
        if not self.rules_enabled:
            return pred_haraka
        char_index = K.argmax(inputs[:, -1], axis=-1)
        # Force the correct haraka on some letters
        forced_diac_chars = {CHAR2INDEX['إ']: 3}
        for f_diac_char, f_diac in forced_diac_chars.items():
            mask = K.reshape(K.cast(K.not_equal(char_index, f_diac_char), 'float32'), (-1, 1))
            pred_haraka = mask * pred_haraka + (1 - mask) * K.one_hot(f_diac, K.int_shape(pred_haraka)[-1])
        # Force the correct haraka before some letters
        f_prev_diac_chars = {CHAR2INDEX['ى']: 1, CHAR2INDEX['ة']: 1}
        prev_char_index = K.argmax(inputs[:, -2], axis=-1)
        for fd_char, f_diac in f_prev_diac_chars.items():
            mask = K.cast(K.not_equal(char_index[1:], fd_char), 'float32')
            mask = K.reshape(K.concatenate([mask, K.ones((1,))], axis=0), (-1, 1))
            pred_haraka = pred_haraka * mask + (1 - mask) * K.one_hot(f_diac, K.int_shape(pred_haraka)[-1])
        # Allow only Fatha, Fathatan, or nothing before ا if it is in the end of the word
        mask = K.reshape(K.concatenate([K.clip(
            K.cast(K.not_equal(char_index[1:-1], CHAR2INDEX['ا']), 'float32') +
            K.cast(K.not_equal(char_index[2:], CHAR2INDEX[' ']), 'float32'), 0, 1), K.ones((2,))], axis=0), (-1, 1))
        pred_haraka = mask * pred_haraka + (1 - mask) * K.constant([1, 1, 0, 0, 0, 1, 0, 0], shape=(1, 8)) * pred_haraka
        # Force Fatha before ا if it is not in the end of the word
        mask = K.reshape(K.concatenate([K.clip(
            K.cast(K.not_equal(char_index[1:-1], CHAR2INDEX['ا']), 'float32') +
            K.cast(K.equal(char_index[2:], CHAR2INDEX[' ']), 'float32'), 0, 1), K.ones((2,))], axis=0), (-1, 1))
        pred_haraka = mask * pred_haraka + (1 - mask) * K.one_hot(1, K.int_shape(pred_haraka)[-1])
        # Force no sukun and tanween at the beginning of the word
        mask = K.reshape(
            K.concatenate([K.zeros((1,)), K.cast(K.not_equal(prev_char_index[1:], CHAR2INDEX[' ']), 'float32')],
                          axis=0), (-1, 1))
        pred_haraka = mask * pred_haraka + (1 - mask) * K.constant([1, 1, 1, 1, 0, 0, 0, 0], shape=(1, 8)) * pred_haraka
        # Allow tanween only at the end of the word
        mask = K.reshape(K.concatenate([K.cast(K.not_equal(char_index[1:], CHAR2INDEX[' ']), 'float32'), K.zeros((1,))],
                                       axis=0), (-1, 1))
        pred_haraka = mask * K.constant([1, 1, 1, 1, 1, 0, 0, 0], shape=(1, 8)) * pred_haraka + (1 - mask) * pred_haraka
        # Prohibit Fathatan on most letters
        mask = K.reshape(K.concatenate([K.clip(
            K.cast(K.not_equal(char_index[1:], CHAR2INDEX[' ']), 'float32') +
            K.cast(K.not_equal(char_index[:-1], CHAR2INDEX['ء']), 'float32'), 0, 1), K.ones((1,))], axis=0), (-1, 1))
        mask *= K.reshape(K.cast(K.not_equal(char_index, CHAR2INDEX['ة']), 'float32'), (-1, 1))
        mask *= K.reshape(K.concatenate([K.clip(
            K.cast(K.not_equal(char_index[1:-1], CHAR2INDEX['ا']), 'float32') +
            K.cast(K.not_equal(char_index[2:], CHAR2INDEX[' ']), 'float32'), 0, 1), K.ones((2,))], axis=0), (-1, 1))
        pred_haraka = mask * K.constant([1, 1, 1, 1, 1, 0, 1, 1], shape=(1, 8)) * pred_haraka + (1 - mask) * pred_haraka
        # Drop haraka from the forbidden characters
        forbidden_chars = [CHAR2INDEX[' '], CHAR2INDEX['0'], CHAR2INDEX['آ'], CHAR2INDEX['ى'], CHAR2INDEX['ا']]
        mask = K.cast(K.not_equal(char_index, forbidden_chars[0]), 'float32')
        for forbidden_char in forbidden_chars[1:]:
            mask *= K.cast(K.not_equal(char_index, forbidden_char), 'float32')
        mask = K.reshape(mask, (-1, 1))
        pred_haraka = mask * pred_haraka + (1 - mask) * K.one_hot(0, K.int_shape(pred_haraka)[-1])
        return pred_haraka

    @staticmethod
    def remove_unwanted_chars(sentences):
        assert isinstance(sentences, Iterable)
        return [' '.join([x for x in WORD_TOKENIZATION_REGEXP.split(NUMBER_REGEXP.sub('0', s))
                          if WORD_TOKENIZATION_REGEXP.match(x)]) for s in sentences]

    CONSONANTS_REGEXP = re.compile('['+''.join(ARABIC_LETTERS - {'ؤ', 'ء', 'ئ', 'أ', 'آ', 'ا', 'ى', 'و', 'ي', 'ة'})+']')

    @staticmethod
    def convert_to_pattern(word):
        assert isinstance(word, str)
        for c in {'ؤ', 'ئ', 'أ', 'آ'}:
            word = word.replace(c, 'ء')
        word = word.replace('ى', 'ا')
        return __class__.CONSONANTS_REGEXP.sub('ح', word)

    def save_history(self, epoch, logs):
        for name in self.values_history.keys():
            self.values_history[name].append(logs[name])
        with open(self.get_history_file_path(), 'wb') as history_file:
            pickle.dump(self.values_history, history_file)

    def train(self, train_sentences, val_sentences, epochs, early_stop_iter):
        print('Removing unwanted characters...')
        train_sentences = self.remove_unwanted_chars(train_sentences)
        val_sentences = self.remove_unwanted_chars(val_sentences)
        print('Generating n-grams...')
        for sentence in train_sentences:
            words = ['<s>'] + sentence.split() + ['<e>']
            undiac_words = [clear_diacritics(w) for w in words]
            for w0_u, w1_d, w1_u, w2_u in zip(undiac_words[:-2], words[1:-1], undiac_words[1:-1], undiac_words[2:]):
                try:
                    self.undiacritized_vocabulary[w1_u].add(w1_d)
                except KeyError:
                    self.undiacritized_vocabulary[w1_u] = {w1_d}
                try:
                    try:
                        self.trigram_context[w0_u, w1_u, w2_u][w1_d] += 1
                    except KeyError:
                        self.trigram_context[w0_u, w1_u, w2_u][w1_d] = 1
                except KeyError:
                    self.trigram_context[w0_u, w1_u, w2_u] = {w1_d: 1}
                try:
                    try:
                        self.bigram_context[w0_u, w1_u][w1_d] += 1
                    except KeyError:
                        self.bigram_context[w0_u, w1_u][w1_d] = 1
                except KeyError:
                    self.bigram_context[w0_u, w1_u] = {w1_d: 1}
                try:
                    self.patterns[self.convert_to_pattern(w1_u)].add(self.convert_to_pattern(w1_d))
                except KeyError:
                    self.patterns[self.convert_to_pattern(w1_u)] = {self.convert_to_pattern(w1_d)}
        with open(self.get_trigrams_file_path(), 'wb') as vocab_file:
            pickle.dump(self.trigram_context, vocab_file)
        with open(self.get_bigrams_file_path(), 'wb') as vocab_file:
            pickle.dump(self.bigram_context, vocab_file)
        with open(self.get_unigrams_file_path(), 'wb') as vocab_file:
            pickle.dump(self.undiacritized_vocabulary, vocab_file)
        with open(self.get_patterns_file_path(), 'wb') as vocab_file:
            pickle.dump(self.patterns, vocab_file)
        print('Processing the dataset...')
        train_ins, train_outs = DiacritizationModel.generate_dataset(train_sentences)
        val_ins, val_outs = DiacritizationModel.generate_dataset(val_sentences)
        print('Calculating parameters...')
        total = 0
        shadda_count = 0
        harakat_counts = np.zeros((8,))
        for shadda_out, harakat_out in train_outs:
            total += len(shadda_out)
            shadda_count += sum(shadda_out)
            for i in set(harakat_out):
                harakat_counts[i] += harakat_out.count(i)
        shadda_weight = (total - shadda_count) / (shadda_count + 1)
        harakat_weights = np.max(harakat_counts) / (harakat_counts + 1)
        self.model.fit_generator(DiacritizedTextDataset(train_ins, train_outs), epochs=epochs,
                                 validation_data=DiacritizedTextDataset(val_ins, val_outs),
                                 class_weight=[{0: 1, 1: shadda_weight}, dict(enumerate(harakat_weights))],
                                 callbacks=[ModelCheckpoint(self.get_weights_file_path(),
                                                            save_weights_only=True, save_best_only=True),
                                            LambdaCallback(on_epoch_end=self.save_history),
                                            EarlyStopping(patience=early_stop_iter, verbose=1),
                                            TerminateOnNaN()], workers=os.cpu_count())

    def test(self, test_sentences, arabic_only, include_no_diacritic):
        # test_ins, test_outs = DiacritizationModel.generate_dataset(self.remove_unwanted_chars(test_sentences))
        # values = self.model.evaluate_generator(DiacritizedTextDataset(test_ins, test_outs))
        # for name, value in zip(self.model.metrics_names, values):
        #     print('{}: {}'.format(name, value))
        print('DER={:.2%} | WER={:.2%} | DERm={:.2%} | WERm={:.2%}'.format(
            *self.der_wer_values(test_sentences, arabic_only, include_no_diacritic)
        ))

    def der_wer_values(self, test_sentences, limit_to_arabic=True, include_no_diacritic=True):
        correct_d, correct_w, total_d, total_w, correct_dm, correct_wm, total_dm = 0, 0, 0, 0, 0, 0, 0
        logging_indexes = set(int(x / 100 * len(test_sentences)) for x in range(1, 101))
        print('Calculating DER and WER values on {} characters'.format('Arabic' if limit_to_arabic else 'all'))
        print('{} no-diacritic Arabic letters'.format('Including' if include_no_diacritic else 'Ignoring'))
        for i, original_sentence in enumerate(test_sentences, 1):
            predicted_sentence = self.diacritize_original(clear_diacritics(original_sentence))
            for orig_word, pred_word in zip(WORD_TOKENIZATION_REGEXP.split(original_sentence),
                                            WORD_TOKENIZATION_REGEXP.split(predicted_sentence)):
                orig_word, pred_word = orig_word.strip(), pred_word.strip()
                if len(orig_word) == 0 or len(pred_word) == 0:  # Rare problematic scenario
                    continue
                if limit_to_arabic:
                    if not WORD_TOKENIZATION_REGEXP.match(orig_word) or NUMBER_REGEXP.match(orig_word):
                        continue
                orig_diacs = np.array([x[::-1] if len(x) == 2 else (x, '') for x in extract_diacritics_2(orig_word)])
                pred_diacs = np.array([x[::-1] if len(x) == 2 else (x, '') for x in extract_diacritics_2(pred_word)])
                if orig_diacs.shape != pred_diacs.shape:  # Rare problematic scenario
                    print('Diacritization mismatch between original and predicted forms: {} {}'.format(orig_word,
                                                                                                       pred_word),
                          file=sys.stderr)
                    continue
                if not include_no_diacritic and WORD_TOKENIZATION_REGEXP.match(orig_word) and\
                        not NUMBER_REGEXP.match(orig_word):
                    diacritics_indexes = orig_diacs[:, 0] != ''
                    pred_diacs = pred_diacs[diacritics_indexes]
                    orig_diacs = orig_diacs[diacritics_indexes]
                correct_w += np.all(orig_diacs == pred_diacs)
                correct_wm += np.all(orig_diacs[:-1] == pred_diacs[:-1])
                total_w += 1
                correct_d += np.sum(np.all(orig_diacs == pred_diacs, axis=1))
                correct_dm += np.sum(np.all(orig_diacs[:-1] == pred_diacs[:-1], axis=1))
                total_d += orig_diacs.shape[0]
                total_dm += orig_diacs[:-1].shape[0]
            if i in logging_indexes:
                print('{}: {}/{} processed ({:.0%}).'.format(datetime.now(), i, len(test_sentences),
                                                             i/len(test_sentences)))
        return 1 - correct_d/total_d, 1 - correct_w/total_w, 1 - correct_dm/total_dm, 1 - correct_wm/total_w

    def save(self):
        self.model.save_weights(self.get_weights_file_path())
        with open(self.get_trigrams_file_path(), 'wb') as trigrams_file:
            pickle.dump(self.trigram_context, trigrams_file)
        with open(self.get_bigrams_file_path(), 'wb') as bigrams_file:
            pickle.dump(self.bigram_context, bigrams_file)
        with open(self.get_unigrams_file_path(), 'wb') as unigrams_file:
            pickle.dump(self.undiacritized_vocabulary, unigrams_file)
        with open(self.get_patterns_file_path(), 'wb') as patterns_file:
            pickle.dump(self.patterns, patterns_file)

    def load(self):
        file_path = self.get_weights_file_path()
        if os.path.isfile(file_path):
            self.model.load_weights(file_path)
        vocab_path = self.get_trigrams_file_path()
        if os.path.isfile(vocab_path) and self.trigrams_enabled:
            with open(vocab_path, 'rb') as vocab_file:
                self.trigram_context = pickle.load(vocab_file)
        vocab_path = self.get_bigrams_file_path()
        if os.path.isfile(vocab_path) and self.bigrams_enabled:
            with open(vocab_path, 'rb') as vocab_file:
                self.bigram_context = pickle.load(vocab_file)
        vocab_path = self.get_unigrams_file_path()
        if os.path.isfile(vocab_path) and self.unigrams_enabled:
            with open(vocab_path, 'rb') as vocab_file:
                self.undiacritized_vocabulary = pickle.load(vocab_file)
        vocab_path = self.get_patterns_file_path()
        if os.path.isfile(vocab_path) and self.patterns_enabled:
            with open(vocab_path, 'rb') as vocab_file:
                self.patterns = pickle.load(vocab_file)

    def diacritize_processed(self, u_p_text):
        assert isinstance(u_p_text, str)
        text_indices = [CHAR2INDEX[x] for x in u_p_text]
        input = add_time_steps(to_categorical(text_indices, len(CHAR2INDEX)), DiacritizationModel.TIME_STEPS, False)
        shadda_pred, harakat_pred = self.model.predict_on_batch(input)
        shaddat = [NAME2DIACRITIC['Shadda'] if x >= 0.5 else '' for x in shadda_pred]
        harakat = [self.index_to_diacritic(np.argmax(x)) for x in harakat_pred]
        d_words = ('<s> ' + ''.join([l + sh + h for l, sh, h in zip(u_p_text, shaddat, harakat)]) + ' <e>').split(' ')
        correct_words = []
        for prev_word, word, next_word in zip(d_words[:-2], d_words[1:-1], d_words[2:]):
            word_u = clear_diacritics(word)
            prev_word_u = clear_diacritics(prev_word)
            next_word_u = clear_diacritics(next_word)
            try:
                best_word = ''
                max_frequency = 0
                for diacritized_word, frequency in self.trigram_context[prev_word_u, word_u, next_word_u].items():
                    if frequency > max_frequency:
                        max_frequency = frequency
                        best_word = diacritized_word
                word = best_word
            except KeyError:  # undiacritized trigram context was not found for this word
                try:
                    best_word = ''
                    max_frequency = 0
                    for diacritized_word, frequency in self.bigram_context[prev_word_u, word_u].items():
                        if frequency > max_frequency:
                            max_frequency = frequency
                            best_word = diacritized_word
                    word = best_word
                except KeyError:  # undiacritized bigram context was not found for this word
                    try:
                        possible_words = list(self.undiacritized_vocabulary[word_u])
                        distances = [self.levenshtein_distance(word, w_d) for w_d in possible_words]
                        word = possible_words[np.argmin(distances)]
                    except KeyError:  # undiacritized word was not found in the dictionary
                        try:
                            u_pattern = self.convert_to_pattern(word_u)
                            d_pattern = self.convert_to_pattern(word)
                            possible_patterns = list(self.patterns[u_pattern])
                            distances = [self.levenshtein_distance(d_pattern, p_d) for p_d in possible_patterns]
                            best_pattern = possible_patterns[np.argmin(distances)]
                            diacritics = extract_diacritics_2(best_pattern)
                            word = ''
                            for l, d in zip(word_u, diacritics):
                                word += l + (d if len(d) < 2 else d[0] + d[1])
                        except KeyError:
                            pass
            correct_words.append(word)
        return ' '.join(correct_words)

    def diacritize_original(self, u_text):
        assert isinstance(u_text, str)
        numbers_words = NUMBER_REGEXP.findall(u_text)
        u_text = NUMBER_REGEXP.sub('0', clear_diacritics(u_text))
        segments = WORD_TOKENIZATION_REGEXP.split(u_text)
        valid_segments = [x for x in segments if WORD_TOKENIZATION_REGEXP.match(x)]
        diacritized_valid_words = self.diacritize_processed(' '.join(valid_segments)).split(' ')
        start_index = 0
        for d_word in diacritized_valid_words:
            u_text = u_text[:start_index] + u_text[start_index:].replace(clear_diacritics(d_word), d_word, 1)
            start_index = max(u_text.index(d_word) + len(d_word), start_index + len(d_word))
        for nw in numbers_words:
            u_text = ZERO_REGEXP.sub(nw, u_text, 1)
        return u_text


class DiacritizedTextDataset(Sequence):

    def __init__(self, in_indices, out_indices):
        self.text_indices = in_indices
        self.diacritics_indices = out_indices

    def __len__(self):
        return len(self.text_indices)

    def __getitem__(self, index):
        input = add_time_steps(to_categorical(self.text_indices[index], len(CHAR2INDEX)),
                               DiacritizationModel.TIME_STEPS, False)
        target_harakat = to_categorical(self.diacritics_indices[index][1], 8)
        target_shadda = np.array(self.diacritics_indices[index][0], dtype=np.float).reshape((-1, 1))
        return input, [target_shadda, target_harakat]
