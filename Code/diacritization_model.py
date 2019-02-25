"""
Module containing the new diacritization model.
"""
import os.path
import pickle
from collections import Iterable

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten, Bidirectional, Input, Layer, Lambda
from tensorflow.keras.metrics import binary_accuracy, categorical_accuracy
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.utils import Sequence, to_categorical

from dataset_preprocessing import NAME2DIACRITIC, CHAR2INDEX, extract_diacritics_2, clear_diacritics, add_time_steps


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

    def __init__(self, save_dir='.'):
        """
        Construct a automatic diacritization system model.
        :param save_dir: the path of the directory containing the weights and the history files.
        """
        self.input_layer = Input(shape=(self.TIME_STEPS, len(CHAR2INDEX)))
        self.inner_layers = [
            Bidirectional(LSTM(64, return_sequences=True, unroll=True)),
            Bidirectional(LSTM(64, return_sequences=True, unroll=True)),
            Conv1D(128, 3, activation='tanh', padding='valid'),
            Flatten(),
            (Dense(8, activation='tanh'), Dense(64, activation='tanh'))
        ]
        previous_layer = self.input_layer
        for layer in self.inner_layers[:-1]:
            previous_layer = layer(previous_layer)
        shadda_side, haraka_side = self.inner_layers[-1]
        shadda_side = shadda_side(previous_layer)
        haraka_side = haraka_side(previous_layer)
        self.output_shadda_layer = Dense(1, activation='sigmoid')(shadda_side)
        self.output_haraka_layer = Dense(8, activation='softmax')(haraka_side)
        self.shadda_corrections_layer = Lambda(self.shadda_post_corrections, name='output_shadda')(
            [self.input_layer, self.output_shadda_layer, self.output_haraka_layer]
        )
        self.haraka_corrections_layer = Lambda(self.haraka_post_corrections, name='output_haraka')(
            [self.input_layer, self.output_haraka_layer, self.output_shadda_layer]
        )
        self.model = Model(inputs=self.input_layer, outputs=[self.shadda_corrections_layer,
                                                             self.haraka_corrections_layer])
        self.model.compile(self.OPTIMIZER,
                           {'output_haraka': 'categorical_crossentropy', 'output_shadda': 'binary_crossentropy'},
                           {'output_haraka': [categorical_accuracy, precision, recall],
                            'output_shadda': [binary_accuracy, precision, recall]})
        self.values_history = dict((k, []) for k in self.model.metrics_names + ['val_'+x for x in
                                                                                self.model.metrics_names])
        self.save_dir = save_dir
        if os.path.isfile(self.get_history_file_path()):
            with open(self.get_history_file_path(), 'rb') as history_file:
                self.values_history = pickle.load(history_file)

    def get_weights_file_path(self):
        layer_shapes = [str(l.output_shape[-1]) if isinstance(l, Layer)
                        else ','.join([str(sl.output_shape[-1]) for sl in l]) for l in self.inner_layers]
        return os.path.join(self.save_dir, type(self).__name__ + '_' + '-'.join(layer_shapes) + '.h5')

    def get_history_file_path(self):
        return self.get_weights_file_path()[:-3]+'_history.pkl'

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

    @staticmethod
    def shadda_post_corrections(in_out):
        """
        Drop any obviously misplaced shadda marks according to the character and its context.
        :param in_out: input layer and prediction layers outputs.
        :return: corrected predictions.
        """
        inputs, pred_shadda, pred_haraka = in_out
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
        allowed_instances *= K.cast(K.not_equal(K.argmax(pred_haraka, axis=1), 4), 'float32') * K.max(pred_haraka,
                                                                                                      axis=1)
        return K.reshape(allowed_instances, (-1, 1)) * pred_shadda

    @staticmethod
    def haraka_post_corrections(in_out):
        """
        Change any obviously wrong haraka marks according to the character and its context.
        :param in_out: input layer and prediction layers outputs.
        :return: corrected predictions.
        """
        inputs, pred_haraka, pred_shadda = in_out
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

    def save_history(self, epoch, logs):
        for name in self.values_history.keys():
            self.values_history[name].append(logs[name])
        with open(self.get_history_file_path(), 'wb') as history_file:
            pickle.dump(self.values_history, history_file)

    def train(self, train_sentences, val_sentences, epochs):
        train_ins, train_outs = DiacritizationModel.generate_dataset(train_sentences)
        val_ins, val_outs = DiacritizationModel.generate_dataset(val_sentences)
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
                                            LambdaCallback(on_epoch_end=self.save_history)], workers=os.cpu_count())

    def test(self, test_sentences):
        test_ins, test_outs = DiacritizationModel.generate_dataset(test_sentences)
        values = self.model.evaluate_generator(DiacritizedTextDataset(test_ins, test_outs))
        for name, value in zip(self.model.metrics_names, values):
            print('{}: {}'.format(name, value))

    def save(self):
        self.model.save_weights(self.get_weights_file_path())

    def load(self):
        file_path = self.get_weights_file_path()
        if os.path.isfile(file_path):
            self.model.load_weights(file_path)

    def diacritize(self, text):
        text_indices = [CHAR2INDEX[x] for x in text]
        input = add_time_steps(to_categorical(text_indices, len(CHAR2INDEX)), DiacritizationModel.TIME_STEPS, False)
        shadda_pred, harakat_pred = self.model.predict_on_batch(input)
        shaddat = [NAME2DIACRITIC['Shadda'] if x >= 0.5 else '' for x in shadda_pred]
        harakat = [self.index_to_diacritic(np.argmax(x)) for x in harakat_pred]
        return ''.join([l+sh+h for l, sh, h in zip(text, shaddat, harakat)])


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


if __name__ == '__main__':
    print('Loading train sentences...')
    train_sents = []
    with open('D:/MSA_dataset_train.txt', 'rt', encoding='utf-8') as dataset_file:
        for line in dataset_file:
            train_sents.append(line.rstrip('\n'))
    print('Loading validation sentences...')
    val_sents = []
    with open('D:/MSA_dataset_val.txt', 'rt', encoding='utf-8') as dataset_file:
        for line in dataset_file:
            val_sents.append(line.rstrip('\n'))
    print('Loading test sentences...')
    test_sents = []
    with open('D:/MSA_dataset_test.txt', 'rt', encoding='utf-8') as dataset_file:
        for line in dataset_file:
            test_sents.append(line.rstrip('\n'))
    print('Making model and training...')
    model = DiacritizationModel()
    model.load()
    # model.train(train_sents, val_sents, 5)
    # os.system('sudo poweroff')
    # model.test(test_sents)
    from random import sample
    for s in sample(test_sents, 10):
        undiacritized = clear_diacritics(s)
        print('_'*len(undiacritized))
        print(model.diacritize(undiacritized))
        print(s)
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(13, 4))
    # loss_axes = plt.subplot(1, 3, 1)
    # loss_axes.plot(np.arange(len(model.values_history['loss']))+1, model.values_history['loss'], label='Train')
    # loss_axes.plot(np.arange(len(model.values_history['loss']))+1, model.values_history['val_loss'], label='Validation')
    # loss_axes.set_title('Loss')
    # loss_axes.legend()
    # shadda_axes = plt.subplot(1, 3, 2)
    # shadda_axes.plot(np.arange(len(model.values_history['loss']))+1, model.values_history['output_shadda_binary_accuracy'], label='Train')
    # shadda_axes.plot(np.arange(len(model.values_history['loss']))+1, model.values_history['val_output_shadda_binary_accuracy'], label='Validation')
    # shadda_axes.set_title('Shadda accuracy')
    # shadda_axes.legend()
    # harakat_axes = plt.subplot(1, 3, 3)
    # harakat_axes.plot(np.arange(len(model.values_history['loss']))+1, model.values_history['output_haraka_categorical_accuracy'], label='Train')
    # harakat_axes.plot(np.arange(len(model.values_history['loss']))+1, model.values_history['val_output_haraka_categorical_accuracy'], label='Validation')
    # harakat_axes.set_title('Harakat accuracy')
    # harakat_axes.legend()
    # plt.show()
