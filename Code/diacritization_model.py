"""
Module containing the new diacritization model.
"""
from collections import Iterable

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Flatten, Bidirectional, Input
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.utils import Sequence, to_categorical

from dataset_preprocessing import NAME2DIACRITIC, CHAR2INDEX, extract_diacritics_2, clear_diacritics, add_time_steps


class DiacritizationModel:

    TIME_STEPS = 10
    OPTIMIZER = Adadelta()

    def __init__(self):
        super().__init__()
        self.input_layer = Input(shape=(self.TIME_STEPS, len(CHAR2INDEX)))
        self.inner_layers = [
            Bidirectional(LSTM(128, return_sequences=True, unroll=True)),
            Bidirectional(LSTM(64, return_sequences=True, unroll=True)),
            Flatten(),
            Dense(32, activation='relu')
        ]
        previous_layer = self.input_layer
        for layer in self.inner_layers:
            previous_layer = layer(previous_layer)
        self.output_shadda_layer = Dense(1, activation='sigmoid', name='output_shadda')(previous_layer)
        self.output_haraka_layer = Dense(8, activation='softmax', name='output_haraka')(previous_layer)
        self.model = Model(inputs=self.input_layer, outputs=[self.output_shadda_layer, self.output_haraka_layer])

    def generate_file_name(self):
        layer_shapes = [str(l.output_shape[-1]) for l in self.inner_layers]
        return type(self).__name__ + '_' + '-'.join(layer_shapes) + '.h5'

    @staticmethod
    def diacritic_to_index(diacritic):
        return ['', NAME2DIACRITIC['Fatha'], NAME2DIACRITIC['Damma'],  NAME2DIACRITIC['Kasra'], NAME2DIACRITIC['Sukun'],
                NAME2DIACRITIC['Fathatan'], NAME2DIACRITIC['Dammatan'], NAME2DIACRITIC['Kasratan']].index(diacritic)

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
                harakat_indices.append(DiacritizationModel.diacritic_to_index(d[-1]) if d and d[-1] != NAME2DIACRITIC['Shadda'] else 0)
            text_indices = [CHAR2INDEX[x] for x in letters_text]
            targets.append((shadda_positions, harakat_indices))
            inputs.append(text_indices)
        return inputs, targets

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
        self.model.compile(self.OPTIMIZER,
                           {'output_haraka': 'categorical_crossentropy', 'output_shadda': 'binary_crossentropy'},
                           {'output_haraka': 'categorical_accuracy', 'output_shadda': 'binary_accuracy'})
        self.model.fit_generator(DiacritizedTextDataset(train_ins, train_outs), epochs=epochs,
                                 validation_data=DiacritizedTextDataset(val_ins, val_outs),
                                 class_weight=[{0: 1, 1: shadda_weight}, dict(enumerate(harakat_weights))])


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
    train_sents = []
    with open('D:/MSA_dataset_train.txt', 'rt', encoding='utf-8') as dataset_file:
        for line in dataset_file:
            train_sents.append(line.rstrip('\n'))
    val_sents = []
    with open('D:/MSA_dataset_val.txt', 'rt', encoding='utf-8') as dataset_file:
        for line in dataset_file:
            val_sents.append(line.rstrip('\n'))
    model = DiacritizationModel()
    model.train(train_sents, val_sents, 1)
