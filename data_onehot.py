# -*- coding: utf-8 -*-
import numpy as np
import sys

from numpy import array
from keras.utils import to_categorical

# 'A': array([1, 0, 0, 0, 0]),
# 'C': array([0, 1, 0, 0, 0]), 
# 'G': array([0, 0, 1, 0, 0]),
# 'T': array([0, 0, 0, 1, 0]), 
# 'E': array([0, 0, 0, 0, 1]), 
alphabet = ['A', 'C', 'G', 'T']


# Get data's max length
def getMaxLength(datasets):
    length_list = []
    for data in datasets:
        length_list.append(len(data))
    return max(length_list)


# Get raw sequences
def getRawSequences(f):
    seqslst = []
    while True:
        s = f.readline()
        if not s:
            break
        else:
            if '>' not in s:
                seq = s.split('\n')[0]
                seqslst.append(seq)
    return seqslst


# Get sparse profile
def getSparseProfile(instances, alphabet, vdim):
    sparse_dict = getSparseDict(alphabet)
    X = []
    for sequence in instances:
        vector = getSparseProfileVector(sequence, sparse_dict, vdim)
        X.append(vector)
    X = array(X, dtype=object)
    return X


# Get sparse dict
def getSparseDict(alphabet):
    alphabet_num = len(alphabet)
    identity_matrix = np.eye(alphabet_num + 1, dtype=int)
    sparse_dict = {alphabet[i]: identity_matrix[i] for i in range(alphabet_num)}
    sparse_dict['E'] = identity_matrix[alphabet_num]
    return sparse_dict


# Get sparse profile vector
def getSparseProfileVector(sequence, sparse_dict, vdim):
    seq_length = len(sequence)
    sequence = sequence + 'E' * (vdim - seq_length) if seq_length <= vdim else sequence[0:vdim]
    vector = sparse_dict.get(sequence[0])
    for i in range(1, vdim):
        temp = sparse_dict.get(sequence[i])
        vector = np.hstack((vector, temp))
    return vector


# Read file
def getOnehotData():
    posi_samples_file = sys.argv[1]
    nega_samples_file = sys.argv[2]
    file_name = posi_samples_file.split('_')[0].split('/')[-1]
    fp = open(posi_samples_file, 'r')
    posis = getRawSequences(fp)
    fn = open(nega_samples_file, 'r')
    negas = getRawSequences(fn)
    instances = array(posis + negas)
    max_length = getMaxLength(instances)
    print('The ' + file_name + ' sequences max length is:', max_length)

    # Sequences
    X = getSparseProfile(instances, alphabet, vdim=max_length)
    # np.savetxt(animal + '_sequence.txt', X, fmt='%s')

    # Label
    Y = array([1] * len(posis) + [0] * len(negas), dtype=int)

    # Onehot encoding
    Y = to_categorical(Y, num_classes=2, dtype='int')
    # np.savetxt(animal + '_label.txt', Y, fmt='%d')

    return X, Y, max_length, file_name
