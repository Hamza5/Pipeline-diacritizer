"""
Module containing the models used for automatic diacritization.
"""

from random import shuffle

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten
from tensorflow.keras.metrics import binary_crossentropy

from dataset_preprocessing import read_text_file, keep_selected_diacritics, extract_diacritics, NAME2DIACRITIC, text_to_one_hot, clear_diacritics, add_time_steps, clean_text


def generate_shadda_dataset(sentences, context_size):
    """
    Generate a dataset for training on shadda only
    :param sentences: list of str, the sentences.
    :param context_size: int, the time steps in the dataset.
    :return: list of  input matrices and list of target matrices, each element is a batch.
    """
    targets = [keep_selected_diacritics(clean_text(s), {NAME2DIACRITIC['Shadda']}) for s in sentences if
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
    model = Sequential([
        LSTM(100, dropout=0.9),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    file_paths = [
        r'D:\Data\Documents\Tashkeela-arabic-diacritized-text-utf8-0.3\texts.txt\إتحاف المهرة لابن حجر.txt',
        r'D:\Data\Documents\Tashkeela-arabic-diacritized-text-utf8-0.3\texts.txt\أحكام القرآن لابن العربي.txt',
        #r'D:\Data\Documents\Tashkeela-arabic-diacritized-text-utf8-0.3\texts.txt\أدب الدنيا والدين.txt',
        #r'D:\Data\Documents\Tashkeela-arabic-diacritized-text-utf8-0.3\texts.txt\الأحكام السلطانية.txt'
    ]
    sentences = []
    for file_path in file_paths:
        sentences += read_text_file(file_path)
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
    train_inputs, train_targets = generate_shadda_dataset(train_sentences, 5)
    print('Generating test dataset...')
    test_inputs, test_targets = generate_shadda_dataset(test_sentences, 5)
    print('Training')
    model.compile('rmsprop', 'binary_crossentropy', [binary_crossentropy, precision, recall])
    for i in range(1, 51):
        print('Iteration', i)
        acc = 0
        loss = 0
        prec = 0
        rec = 0
        for k in range(len(train_targets)):
            l, a, p, r = model.train_on_batch(train_inputs[k], train_targets[k], class_weight={0: 1, 1: balancing_factor})
            acc += a
            loss += l
            prec += p
            rec += r
            if k % 1000 == 0:
                print('Train:')
                print('Loss = {:.5f} | Accuracy = {:.2%} | Precision = {:.2%} | Recall = {:.2%}'.format(loss/k, acc/k,
                                                                                                        prec/k, rec/k))
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
        print('Test:')
        print('Loss = {:.5f} | Accuracy = {:.2%} | Precision = {:.2%} | Recall = {:.2%}'.format(
            loss/len(test_targets), acc/len(test_targets), prec/len(test_targets), rec/len(test_targets))
        )
