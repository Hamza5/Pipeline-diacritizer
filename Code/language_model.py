"""
Module containing the models used for automatic diacritization.
"""
import numpy as np
from random import shuffle, sample

import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Lambda, Input
from tensorflow.keras import metrics
from tensorflow.keras import losses

from dataset_preprocessing import keep_selected_diacritics, NAME2DIACRITIC, clear_diacritics, extract_diacritics, \
    add_time_steps, text_to_one_hot, read_text_file, filter_tokenized_sentence, tokenize, fix_double_diacritics_error, \
    input_to_sentence, merge_diacritics, CHAR2INDEX


TIME_STEPS = 5


def generate_shadda_dataset(sentences, context_size):
    """
    Generate a dataset for training on shadda only
    :param sentences: list of str, the sentences.
    :param context_size: int, the time steps in the dataset.
    :return: list of  input matrices and list of target matrices, each element is a batch.
    """
    targets = [keep_selected_diacritics(s, {NAME2DIACRITIC['Shadda']}) for s in sentences if
               NAME2DIACRITIC['Shadda'] in s]
    input_array = []
    target_array = []
    for target in targets:
        u_target = clear_diacritics(target)
        only_shadda_labels = extract_diacritics(target)
        target_labels = np.zeros((len(u_target)))
        target_labels[np.array(only_shadda_labels) == NAME2DIACRITIC['Shadda']] = 1
        input_matrix = add_time_steps(text_to_one_hot(u_target), context_size)
        input_array.append(input_matrix)
        target_array.append(target_labels)
    return input_array, target_array


def keras_precision(y_true, y_pred):
    """Precision metric.	
    Only computes a batch-wise average of precision. Computes the precision, a
    metric for multi-label classification of how many selected items are
    relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def keras_recall(y_true, y_pred):
    """Recall metric.	
    Only computes a batch-wise average of recall. Computes the recall, a metric
    for multi-label classification of how many relevant items are selected.	
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def shadda_post_corrections(in_out):
    inputs, predictions = in_out
    last_char = inputs[:, -1]
    keep_value = K.cast(K.not_equal(K.argmax(last_char, axis=-1), CHAR2INDEX[' ']), 'float32')
    return K.reshape(keep_value, (-1, 1)) * predictions


if __name__ == '__main__':
    file_paths = [
        r'D:\Data\Documents\Tashkeela-arabic-diacritized-text-utf8-0.3\texts.txt\إتحاف المهرة لابن حجر.txt',
        r'D:\Data\Documents\Tashkeela-arabic-diacritized-text-utf8-0.3\texts.txt\أحكام القرآن لابن العربي.txt',
        r'D:\Data\Documents\Tashkeela-arabic-diacritized-text-utf8-0.3\texts.txt\أدب الدنيا والدين.txt',
        r'D:\Data\Documents\Tashkeela-arabic-diacritized-text-utf8-0.3\texts.txt\الأحكام السلطانية.txt'
    ]
    sentences = []
    print('Loading text...')
    for file_path in file_paths:
        sentences += read_text_file(file_path)
    print('Parsing and cleaning...')
    sentences = [' '.join(sf) for sf in
                 filter(lambda x: len(x) > 0, [filter_tokenized_sentence(tokenize(fix_double_diacritics_error(s)))
                                               for s in sentences])]
    shuffle(sentences)
    print('Number of sentences =', len(sentences))
    train_size = round(0.9 * len(sentences))
    train_sentences = sentences[:train_size]
    test_sentences = sentences[train_size:]
    print('In train =', len(train_sentences))
    print('In test =', len(test_sentences))
    shadda_count = 0
    total = 0
    for sentence in train_sentences:
        total += len(clear_diacritics(sentence))
        shadda_count += len(list(filter(lambda x: x == NAME2DIACRITIC['Shadda'], sentence)))
    balancing_factor = total/shadda_count
    print('Balancing factor = {:.5f}'.format(balancing_factor))
    print('Generating train dataset...')
    train_inputs, train_targets = generate_shadda_dataset(train_sentences, TIME_STEPS)
    print('Generating test dataset...')
    test_inputs, test_targets = generate_shadda_dataset(test_sentences, TIME_STEPS)
    print('Training...')
    input_layer = Input(shape=(TIME_STEPS, len(CHAR2INDEX)))
    lstm_layer = LSTM(100, dropout=0.9)(input_layer)
    dense_layer = Dense(1, activation='sigmoid')(lstm_layer)
    post_layer = Lambda(shadda_post_corrections)([input_layer, dense_layer])
    model = Model(inputs=input_layer, outputs=post_layer)
    model.compile('rmsprop', losses.binary_crossentropy, [metrics.binary_accuracy, keras_precision, keras_recall])
    for i in range(1, 21):
        print('Iteration', i)
        acc = 0
        loss = 0
        prec = 0
        rec = 0
        for k in range(len(train_targets)):
            l, a, p, r = model.train_on_batch(train_inputs[k], train_targets[k],
                                              class_weight={0: 1, 1: balancing_factor})
            acc += a
            loss += l
            prec += p
            rec += r
            if k % 1000 == 0:
                print('Train ({}/{}):'.format(k+1, len(train_targets)))
                print('Loss = {:.5f} | Accuracy = {:.2%} | Precision = {:.2%} | Recall = {:.2%}'.format(
                    loss/(k+1), acc/(k+1), prec/(k+1), rec/(k+1))
                )
        print('Test:')
        acc = 0
        loss = 0
        prec = 0
        rec = 0
        for k in range(len(test_targets)):
            l, a, p, r = model.test_on_batch(test_inputs[k], test_targets[k])
            acc += a
            loss += l
            prec += p
            rec += r
        print('Loss = {:.5f} | Accuracy = {:.2%} | Precision = {:.2%} | Recall = {:.2%}'.format(
            loss/len(test_targets), acc/len(test_targets), prec/len(test_targets), rec/len(test_targets))
        )
        print('Test predictions samples:')
        for k in sample(range(len(test_targets)), 10):
            predicted_indices = model.predict_on_batch(test_inputs[k]) >= 0.5
            u_text = input_to_sentence(test_inputs[k])
            diacritics = [NAME2DIACRITIC['Shadda'] if c else '' for c in predicted_indices]
            print(merge_diacritics(u_text, diacritics))
