"""
Module containing the models used for automatic diacritization.
"""
import os.path
import re
from abc import ABC, abstractmethod
from numbers import Real
from random import sample
from typing import Collection

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras import callbacks
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras import utils
from tensorflow.keras.layers import LSTM, Dense, Lambda, Input, Bidirectional

from dataset_preprocessing import keep_selected_diacritics, NAME2DIACRITIC, clear_diacritics, extract_diacritics, \
    text_to_indices, CHAR2INDEX, ARABIC_DIACRITICS, add_time_steps, input_to_sentence, merge_diacritics

LAST_DIACRITIC_REGEXP = re.compile('['+''.join(ARABIC_DIACRITICS)+r']+(?= |$)')


class DiacritizationModel(ABC):

    DEFAULT_TIME_STEPS = 10
    DEFAULT_OPTIMIZER = optimizers.Adam()

    @staticmethod
    @abstractmethod
    def generate_dataset(sentences):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def post_corrections(in_out):
        raise NotImplementedError

    def __init__(self, lstm_sizes, dropouts, output_size, save_dir):
        self.train_inputs = []
        self.train_targets = []
        self.val_inputs = []
        self.val_targets = []
        self.balancing_factors = None
        self.save_dir = save_dir
        self.time_steps = self.DEFAULT_TIME_STEPS
        self.optimizer = self.DEFAULT_OPTIMIZER
        self.metrics = [metrics.binary_accuracy, precision, recall]
        self.model = self._build_model(lstm_sizes, dropouts, output_size)

    def _build_model(self, lstm_layers_sizes, dropouts, output_layer_size):
        assert len(lstm_layers_sizes) == len(dropouts) and len(lstm_layers_sizes) > 0
        last_layer = input_layer = Input(shape=(self.time_steps, len(CHAR2INDEX)))
        for layer_size, dropout_factor in zip(lstm_layers_sizes[:-1], dropouts[:-1]):
            last_layer = Bidirectional(LSTM(layer_size, dropout=dropout_factor, return_sequences=True,
                                            unroll=True))(last_layer)
        last_layer = Bidirectional(LSTM(lstm_layers_sizes[-1], dropout=dropouts[-1], unroll=True))(last_layer)
        output_layer = Dense(output_layer_size,
                             activation=('sigmoid' if output_layer_size == 1 else 'softmax'))(last_layer)
        post_corrections_layer = Lambda(self.post_corrections)([input_layer, output_layer])
        model = Model(inputs=input_layer, outputs=post_corrections_layer)
        model.compile(self.optimizer,
                      losses.binary_crossentropy if output_layer_size == 1 else losses.categorical_crossentropy,
                      self.metrics)
        return model

    def _generate_file_name(self):
        layer_shapes = [str(l.output_shape[-1]) for l in self.model.layers]
        return str(self.__class__).split('.')[-1].rstrip("'>") + '_' + '-'.join(layer_shapes) + '.h5'

    def save(self, directory_path='.'):
        assert isinstance(directory_path, str)
        self.model.save_weights(os.path.join(directory_path, self._generate_file_name()))

    def load(self, directory_path='.'):
        assert isinstance(directory_path, str)
        file_path = os.path.join(directory_path, self._generate_file_name())
        if os.path.exists(file_path):
            self.model.load_weights(file_path)

    def feed_data(self, train_sentences, val_sentences):
        print('Generating train dataset...')
        self.train_inputs, self.train_targets = self.generate_dataset(train_sentences)
        print('Generating validation dataset...')
        self.val_inputs, self.val_targets = self.generate_dataset(val_sentences)
        print('Generated {} train batches and {} validation batches.'.format(len(self.train_targets),
                                                                             len(self.val_targets)))
        self.balancing_factors = self.calculate_balancing_factors()

    def train(self, epochs, word_level):
        self.load(self.save_dir)
        for i in range(1, epochs + 1):
            acc = 0
            loss = 0
            prec = 0
            rec = 0
            sum_factors = 0
            for k in range(len(self.train_targets)):
                target = self.train_targets[k]
                if len(target.shape) > 1:
                    target = utils.to_categorical(self.train_targets[k], target.shape[-1])
                l, a, p, r = self.model.train_on_batch(
                    add_time_steps(utils.to_categorical(self.train_inputs[k], len(CHAR2INDEX)),
                                   self.time_steps, word_level),
                    target, class_weight=self.balancing_factors
                )
                acc += a * self.train_targets[k].shape[0]
                loss += l * self.train_targets[k].shape[0]
                prec += p * self.train_targets[k].shape[0]
                rec += r * self.train_targets[k].shape[0]
                sum_factors += self.train_targets[k].shape[0]
                if k % 1000 == 0:
                    print('{}/{}: Train ({}/{}):'.format(i, epochs, k + 1, len(self.train_targets)))
                    print('Loss = {:.5f} | Accuracy = {:.2%} | Precision = {:.2%} | Recall = {:.2%}'.format(
                        loss / sum_factors, acc / sum_factors, prec / sum_factors, rec / sum_factors)
                    )
            print('{}/{}: Validation:'.format(i, epochs))
            acc = 0
            loss = 0
            prec = 0
            rec = 0
            sum_factors = 0
            for k in range(len(self.val_targets)):
                target = self.val_targets[k]
                if len(target.shape) > 1:
                    target = utils.to_categorical(self.val_targets[k], target.shape[-1])
                l, a, p, r = self.model.test_on_batch(
                    add_time_steps(utils.to_categorical(self.val_inputs[k], len(CHAR2INDEX)),
                                   self.time_steps, word_level),
                    target
                )
                acc += a * self.val_targets[k].shape[0]
                loss += l * self.val_targets[k].shape[0]
                prec += p * self.val_targets[k].shape[0]
                rec += r * self.val_targets[k].shape[0]
                sum_factors += self.val_targets[k].shape[0]
            print('Loss = {:.5f} | Accuracy = {:.2%} | Precision = {:.2%} | Recall = {:.2%}'.format(
                loss / sum_factors, acc / sum_factors, prec / sum_factors, rec / sum_factors)
            )
            der = self.calculate_der()
            if isinstance(der, Real):
                print('DER = {:.2%}'.format(der), end='  ')
            wer = self.calculate_wer()
            if isinstance(wer, Real):
                print('WER = {:.2%}'.format(wer))
            else:
                print()
            self.save(self.save_dir)

    def test(self, test_sentences, word_level):
        print('Generating test dataset...')
        test_inputs, test_targets = self.generate_dataset(test_sentences)
        print('Test:')
        acc = 0
        loss = 0
        prec = 0
        rec = 0
        sum_factors = 0
        for k in range(len(test_targets)):
            target = test_targets[k]
            if len(target.shape) > 1:
                target = utils.to_categorical(test_targets[k], target.shape[-1])
            l, a, p, r = self.model.test_on_batch(
                add_time_steps(utils.to_categorical(test_inputs[k], len(CHAR2INDEX)),
                               self.time_steps, word_level),
                target
            )
            acc += a * test_targets[k].shape[0]
            loss += l * test_targets[k].shape[0]
            prec += p * test_targets[k].shape[0]
            rec += r * test_targets[k].shape[0]
            sum_factors += test_targets[k].shape[0]
        print('Loss = {:.5f} | Accuracy = {:.2%} | Precision = {:.2%} | Recall = {:.2%}'.format(
            loss / sum_factors, acc / sum_factors, prec / sum_factors, rec / sum_factors)
        )

    @abstractmethod
    def calculate_balancing_factors(self):
        raise NotImplementedError

    @abstractmethod
    def visualize(self, num_samples):
        """
        Diacritize and show random samples from the validation set.
        :param num_samples: number of examples to show.
        """
        raise NotImplementedError

    def calculate_der(self):
        """
        Calculate the diacritization error rate.
        :return: float, the error rate at diacritics level.
        """
        return NotImplemented

    def calculate_wer(self):
        """
        Calculate the word error rate.
        :return: float, the error rate at word level.
        """
        return NotImplemented


class GeminationModel(DiacritizationModel):

    @staticmethod
    def generate_dataset(sentences):
        """
        Generate a dataset for training on shadda only.
        :param sentences: list of str, the sentences.
        :return: list of input arrays and list of target arrays, each element is a batch.
        """
        assert isinstance(sentences, Collection) and all(isinstance(s, str) for s in sentences)
        targets = [keep_selected_diacritics(s, {NAME2DIACRITIC['Shadda']}) for s in sentences]
        input_array = []
        target_array = []
        for target in targets:
            u_target = clear_diacritics(target)
            only_shadda_labels = extract_diacritics(target)
            target_labels = np.zeros((len(u_target)))
            target_labels[np.array(only_shadda_labels) == NAME2DIACRITIC['Shadda']] = 1
            input_array.append(text_to_indices(u_target))
            target_array.append(target_labels)
        return input_array, target_array

    @staticmethod
    def post_corrections(in_out):
        """
        Correct any obviously misplaced shadda marks according to the character and its context.
        :param in_out: input layer and prediction layer outputs.
        :return: corrected predictions.
        """
        inputs, predictions = in_out
        # Drop the shadda from the forbidden letters
        forbidden_chars = [CHAR2INDEX[' '], CHAR2INDEX['ا'], CHAR2INDEX['ء'], CHAR2INDEX['أ'], CHAR2INDEX['إ'],
                           CHAR2INDEX['آ'], CHAR2INDEX['ى'], CHAR2INDEX['ئ'], CHAR2INDEX['ة'], CHAR2INDEX['0']]
        char_index = K.argmax(inputs[:, -1], axis=-1)
        allowed_instances = K.cast(K.not_equal(char_index, forbidden_chars[0]), 'float32')
        for char in forbidden_chars[1:]:
            allowed_instances *= K.cast(K.not_equal(char_index, char), 'float32')
        allowed_instances *= K.sum(inputs[:, -2], axis=-1)
        # Drop the shadda from the letter following the space
        previous_char_index = K.argmax(inputs[:, -2], axis=-1)
        allowed_instances *= K.cast(K.not_equal(previous_char_index, CHAR2INDEX[' ']), 'float32')
        return K.reshape(allowed_instances, (-1, 1)) * predictions

    def __init__(self, lstm_sizes, dropouts, save_dir):
        super().__init__(lstm_sizes, dropouts, 1, save_dir)

    def train(self, epochs, **kwargs):
        super(GeminationModel, self).train(epochs, False)

    def test(self, test_sentences, **kwargs):
        super(GeminationModel, self).test(test_sentences, False)

    def visualize(self, num_samples):
        print('Validation predictions samples:')
        for k in sample(range(len(self.val_targets)), num_samples):
            val_input = add_time_steps(
                utils.to_categorical(self.val_inputs[k], len(CHAR2INDEX)), self.time_steps, False
            )
            predicted_indices = self.model.predict_on_batch(val_input) >= 0.5
            u_text = input_to_sentence(val_input, False)
            p_diacritics = [NAME2DIACRITIC['Shadda'] if c else '' for c in predicted_indices]
            r_diacritics = [NAME2DIACRITIC['Shadda'] if c else '' for c in self.val_targets[k]]
            print(merge_diacritics(u_text, p_diacritics))
            print(merge_diacritics(u_text, r_diacritics))

    def calculate_balancing_factors(self):
        shadda_count = 0
        total = 0
        for batch_targets in self.train_targets:
            shadda_count += np.sum(batch_targets)
            total += np.shape(batch_targets)[0]
        balancing_factor = total / (shadda_count + 1)
        return {0: 1, 1: balancing_factor}

    def calculate_der(self):
        not_correct = 0
        total = 0
        for l_input, l_target in zip(self.val_inputs, self.val_targets):
            val_input = add_time_steps(utils.to_categorical(l_input, len(CHAR2INDEX)), self.time_steps, False)
            val_prediction = self.model.predict_on_batch(val_input).flatten()
            not_spaces = l_input != CHAR2INDEX[' ']
            total += np.sum(not_spaces)
            not_correct += np.sum(l_target[not_spaces] != np.round(val_prediction[not_spaces]))
        return not_correct / total

    def calculate_wer(self):
        not_correct = 0
        total = 0
        for l_input, l_target in zip(self.val_inputs, self.val_targets):
            val_input = add_time_steps(utils.to_categorical(l_input, len(CHAR2INDEX)), self.time_steps, False)
            val_pred = self.model.predict_on_batch(val_input).flatten()
            spaces_indices = np.concatenate((np.nonzero(l_input == CHAR2INDEX[' '])[0], np.shape(l_input)[0:1]))
            last = 0
            for i in spaces_indices:
                not_correct += not np.all(l_target[last:i] == np.round(val_pred[last:i]))
                total += 1
                last = i + 1
        return not_correct / total


class MorphologicalDiacriticsModel(DiacritizationModel):

    @staticmethod
    def generate_dataset(sentences):
        """
        Generate a dataset for training on non ending diacritics only.
        :param sentences: list of str, the sentences.
        :return: list of input arrays and list of target arrays, each element is a batch.
        """
        harakat = set(NAME2DIACRITIC[x] for x in ('Fatha', 'Damma', 'Kasra', 'Sukun'))
        targets = [LAST_DIACRITIC_REGEXP.sub('', keep_selected_diacritics(s, harakat)) for s in sentences]
        input_array = []
        target_array = []
        for target in targets:
            u_target = clear_diacritics(target)
            harakat_labels = extract_diacritics(target)
            target_labels = np.zeros((len(u_target)))
            target_labels[np.array(harakat_labels) == NAME2DIACRITIC['Fatha']] = 1
            target_labels[np.array(harakat_labels) == NAME2DIACRITIC['Damma']] = 2
            target_labels[np.array(harakat_labels) == NAME2DIACRITIC['Kasra']] = 3
            target_labels[np.array(harakat_labels) == NAME2DIACRITIC['Sukun']] = 4
            input_array.append(text_to_indices(u_target))
            target_array.append(target_labels)
        return input_array, target_array

    @staticmethod
    def post_corrections(in_out):
        """
        Correct any obviously misplaced morphological diacritics marks according to the character and its context.
        :param in_out: input layer and prediction layer outputs.
        :return: corrected predictions.
        """
        inputs, predictions = in_out
        # Drop diacritics from the forbidden letters
        forbidden_chars = [CHAR2INDEX[' '], CHAR2INDEX['آ'], CHAR2INDEX['ى'], CHAR2INDEX['0']]
        char_index = K.argmax(inputs[:, -1], axis=-1)
        mask = K.cast(K.not_equal(char_index, forbidden_chars[0]), 'float32')
        for forbidden_char in forbidden_chars[1:]:
            mask *= K.cast(K.not_equal(char_index, forbidden_char), 'float32')
        mask = K.reshape(mask, (-1, 1))
        predictions = mask * predictions + (1 - mask) * K.one_hot(0, K.int_shape(predictions)[-1])
        # Force the correct diacritic on some letters
        forced_diac_chars = {CHAR2INDEX['إ']: 3}
        for f_diac_char, f_diac in forced_diac_chars.items():
            mask = K.reshape(K.cast(K.not_equal(char_index, f_diac_char), 'float32'), (-1, 1))
            predictions = mask * predictions + (1 - mask) * K.one_hot(f_diac, K.int_shape(predictions)[-1])
        # Force the correct diacritics before some long vowels
        f_prev_diac_chars = {CHAR2INDEX['ا']: 1, CHAR2INDEX['ى']: 1, CHAR2INDEX['ة']: 1}
        prev_char_index = K.argmax(inputs[:, -2], axis=-1)
        for fd_char, f_diac in f_prev_diac_chars.items():
            mask = K.clip(K.cast(K.not_equal(char_index[1:-1], fd_char), 'float32') +
                          K.cast(K.equal(prev_char_index[1:-1], CHAR2INDEX[' ']), 'float32'), 0, 1)
            if fd_char == CHAR2INDEX['ا']:
                mask = K.clip(mask + K.cast(K.equal(char_index[2:], CHAR2INDEX[' ']), 'float32'), 0, 1)
            mask = K.reshape(K.concatenate([mask, K.ones((2,))], axis=0), (-1, 1))
            predictions = predictions * mask + (1 - mask) * K.one_hot(f_diac, K.int_shape(predictions)[-1])
        # Drop the last diacritic from every word
        mask = K.reshape(K.concatenate([K.cast(K.not_equal(char_index[1:], CHAR2INDEX[' ']), 'float32'), K.zeros((1,))],
                                       axis=0), (-1, 1))
        predictions = mask * predictions + (1 - mask) * K.one_hot(0, K.int_shape(predictions)[-1])
        # Force no sukun at the beginning of the word
        mask = K.reshape(
            K.concatenate([K.zeros((1,)), K.cast(K.not_equal(prev_char_index[1:], CHAR2INDEX[' ']), 'float32')]
                          , axis=0), (-1, 1))
        predictions = mask * predictions + (1 - mask) * K.constant([1, 1, 1, 1, 0], shape=(1, 5)) * predictions
        return predictions

    def __init__(self, lstm_sizes, dropouts, save_dir):
        super().__init__(lstm_sizes, dropouts, 5, save_dir)

    def train(self, epochs, **kwargs):
        super(MorphologicalDiacriticsModel, self).train(epochs, False)

    def test(self, test_sentences, **kwargs):
        super(MorphologicalDiacriticsModel, self).test(test_sentences, False)

    def calculate_balancing_factors(self):
        b_factors = np.zeros((5,))
        for tsl in self.val_targets:
            for label in set(tsl):
                for i in range(b_factors.shape[0]):
                    b_factors[i] += np.sum(label == i)
        b_factors = np.max(b_factors) / (b_factors + 1)
        return dict(enumerate(b_factors))

    def visualize(self, num_samples):
        print('Validation predictions samples:')
        for k in sample(range(len(self.val_targets)), num_samples):
            val_input = add_time_steps(
                utils.to_categorical(self.val_inputs[k], len(CHAR2INDEX)), self.time_steps, False
            )
            predicted_indices = np.argmax(self.model.predict_on_batch(val_input), axis=-1)
            u_text = input_to_sentence(val_input, False)
            p_diacritics = np.empty((len(u_text),), dtype=str)
            p_diacritics[predicted_indices == 0] = ''
            p_diacritics[predicted_indices == 1] = NAME2DIACRITIC['Fatha']
            p_diacritics[predicted_indices == 2] = NAME2DIACRITIC['Damma']
            p_diacritics[predicted_indices == 3] = NAME2DIACRITIC['Kasra']
            p_diacritics[predicted_indices == 4] = NAME2DIACRITIC['Sukun']
            r_diacritics = np.empty((len(u_text),), dtype=str)
            r_diacritics[self.val_targets[k] == 0] = ''
            r_diacritics[self.val_targets[k] == 1] = NAME2DIACRITIC['Fatha']
            r_diacritics[self.val_targets[k] == 2] = NAME2DIACRITIC['Damma']
            r_diacritics[self.val_targets[k] == 3] = NAME2DIACRITIC['Kasra']
            r_diacritics[self.val_targets[k] == 4] = NAME2DIACRITIC['Sukun']
            print(merge_diacritics(u_text, p_diacritics.tolist()))
            print(merge_diacritics(u_text, r_diacritics.tolist()))


class LastDiacriticModel(DiacritizationModel):

    @staticmethod
    def generate_dataset(sentences):
        """
        Generate a dataset for training on last diacritics only.
        :param sentences: list of str, the sentences.
        :return: list of input arrays and list of target arrays, each element is a batch.
        """
        inputs = []
        targets = []
        for sentence in sentences:
            target = []
            input = []
            for word in sentence.split():
                if word[-1] in ARABIC_DIACRITICS - {NAME2DIACRITIC['Shadda']}:
                    if word[-1] == NAME2DIACRITIC['Fatha']:
                        target.append(1)
                    elif word[-1] == NAME2DIACRITIC['Damma']:
                        target.append(2)
                    elif word[-1] == NAME2DIACRITIC['Kasra']:
                        target.append(3)
                    elif word[-1] == NAME2DIACRITIC['Sukun']:
                        target.append(4)
                    elif word[-1] == NAME2DIACRITIC['Fathatan']:
                        target.append(5)
                    elif word[-1] == NAME2DIACRITIC['Dammatan']:
                        target.append(6)
                    elif word[-1] == NAME2DIACRITIC['Kasratan']:
                        target.append(7)
                elif word[-1] == 'ا':
                    if len(word) > 1 and word[-2] == NAME2DIACRITIC['Fathatan']:
                        target.append(5)
                    elif len(word) > 1 and word[-2] == NAME2DIACRITIC['Fatha']:
                        target.append(1)
                    else:
                        target.append(0)
                else:
                    target.append(0)
                input.append(np.array([CHAR2INDEX[x] for x in clear_diacritics(word) + ' ']))
            targets.append(np.array(target))
            inputs.append(np.concatenate(input)[:-1])
        return inputs, targets

    @staticmethod
    def post_corrections(in_out):
        """
        Correct any obviously misplaced last diacritic mark according to the last character and its context.
        :param in_out: input layer and prediction layer outputs.
        :return: corrected predictions.
        """
        inputs, predictions = in_out
        last_letter_index = K.argmax(inputs[:, -1], axis=-1)
        # Only Fathatan or Fatha or nothing for the last alef.
        mask = K.reshape(K.cast(K.not_equal(last_letter_index, CHAR2INDEX['ا']), 'float32'), (-1, 1))
        predictions = mask * predictions + (1 - mask) * K.constant([1, 1, 0, 0, 0, 1, 0, 0], shape=(1, 8)) * predictions
        # Nothing for alef maqsura
        mask = K.reshape(K.cast(K.not_equal(last_letter_index, CHAR2INDEX['ى']), 'float32'), (-1, 1))
        predictions = mask * predictions + (1 - mask) * K.constant([1, 0, 0, 0, 0, 0, 0, 0], shape=(1, 8))
        return predictions

    def __init__(self, lstm_sizes, dropouts, save_dir):
        super().__init__(lstm_sizes, dropouts, 8, save_dir)

    def train(self, epochs, **kwargs):
        super(LastDiacriticModel, self).train(epochs, True)

    def test(self, test_sentences, **kwargs):
        super(LastDiacriticModel, self).test(test_sentences, True)

    def calculate_balancing_factors(self):
        b_factors = np.zeros((8,))
        for tsl in self.val_targets:
            for label in set(tsl):
                for i in range(b_factors.shape[0]):
                    b_factors[i] += np.sum(label == i)
        b_factors = np.max(b_factors) / (b_factors + 1)
        return dict(enumerate(b_factors))

    def visualize(self, num_samples):
        print('Validation predictions samples:')
        for k in sample(range(len(self.val_targets)), num_samples):
            val_input = add_time_steps(
                utils.to_categorical(self.val_inputs[k], len(CHAR2INDEX)), self.time_steps, True
            )
            predicted_indices = np.argmax(self.model.predict_on_batch(val_input), axis=-1)
            p_diacritics = np.empty((predicted_indices.shape[0],), dtype=str)
            r_diacritics = np.empty(self.val_targets[k].shape, dtype=str)
            p_diacritics[predicted_indices == 0] = ''
            p_diacritics[predicted_indices == 1] = NAME2DIACRITIC['Fatha']
            p_diacritics[predicted_indices == 2] = NAME2DIACRITIC['Damma']
            p_diacritics[predicted_indices == 3] = NAME2DIACRITIC['Kasra']
            p_diacritics[predicted_indices == 4] = NAME2DIACRITIC['Sukun']
            p_diacritics[predicted_indices == 5] = NAME2DIACRITIC['Fathatan']
            p_diacritics[predicted_indices == 6] = NAME2DIACRITIC['Dammatan']
            p_diacritics[predicted_indices == 7] = NAME2DIACRITIC['Kasratan']
            r_diacritics[self.val_targets[k] == 0] = ''
            r_diacritics[self.val_targets[k] == 1] = NAME2DIACRITIC['Fatha']
            r_diacritics[self.val_targets[k] == 2] = NAME2DIACRITIC['Damma']
            r_diacritics[self.val_targets[k] == 3] = NAME2DIACRITIC['Kasra']
            r_diacritics[self.val_targets[k] == 4] = NAME2DIACRITIC['Sukun']
            r_diacritics[self.val_targets[k] == 5] = NAME2DIACRITIC['Fathatan']
            r_diacritics[self.val_targets[k] == 6] = NAME2DIACRITIC['Dammatan']
            r_diacritics[self.val_targets[k] == 7] = NAME2DIACRITIC['Kasratan']
            u_text = input_to_sentence(val_input, True)
            p_diacritized_sentence = ''
            r_diacritized_sentence = ''
            for word, p_diacritic, r_diacritic in zip(u_text.split(), p_diacritics, r_diacritics):
                if word[-1] == 'ا':
                    p_diacritized_sentence += word[:-1] + p_diacritic + word[-1] + ' '
                    r_diacritized_sentence += word[:-1] + r_diacritic + word[-1] + ' '
                else:
                    p_diacritized_sentence += word + p_diacritic + ' '
                    r_diacritized_sentence += word + r_diacritic + ' '
            print(p_diacritized_sentence[:-1])
            print(r_diacritized_sentence[:-1])


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


if __name__ == '__main__':
    sentences = []
    with open('D:\preprocesed_dataset_test.txt', encoding='UTF-8') as dataset_file:
        for s in dataset_file:
            sentences.append(s.rstrip('\n'))
    print('Number of sentences =', len(sentences))
    model = GeminationModel([128, 64], [0.1, 0.1], '.')
    model.load()
    model.test(sentences)
