"""
Module containing the new diacritization model
"""
from collections import Iterable

from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Flatten, Bidirectional, Input, Dropout

from dataset_preprocessing import NAME2DIACRITIC, CHAR2INDEX, extract_diacritics_2, clear_diacritics


class DiacritizationModel(Model):

    TIME_STEPS = 10

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        self.input_layer = Input(shape=(self.TIME_STEPS, len(CHAR2INDEX)))
        self.inner_layers = [
            Bidirectional(LSTM(128, return_sequences=True, unroll=True)),
            Bidirectional(LSTM(64, return_sequences=True, unroll=True)),
            Flatten(),
            Dense(32, activation='relu')
        ]
        self.output_shadda_layer = Dense(1, activation='sigmoid')
        self.output_haraka_layer = Dense(8, activation='softmax')

    def call(self, inputs, training=None):
        x = self.input_layer(inputs)
        for layer in self.inner_layers:
            if not isinstance(layer, Dropout) or training:
                x = layer(x)
        return [self.output_haraka_layer(x), self.output_shadda_layer(x)]

    def generate_file_name(self):
        layer_shapes = [str(l.output_shape[-1]) for l in self.inner_layers]
        return type(self).__name__ + '_' + '-'.join(layer_shapes) + '.h5'

    @staticmethod
    def diacritic_to_index(diacritic):
        return ['', NAME2DIACRITIC['Fatha'], NAME2DIACRITIC['Damma'],  NAME2DIACRITIC['Kasra'], NAME2DIACRITIC['Sukun'], NAME2DIACRITIC['Fathatan'], NAME2DIACRITIC['Dammatan'], NAME2DIACRITIC['Kasratan']].index(diacritic)

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
