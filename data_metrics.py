from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import keras
import numpy as np
import math
import win_unicode_console

win_unicode_console.enable()

aupr_average_list = []
auc_average_list = []
acc_average_list = []
f1_average_list = []
precision_average_list = []
recall_average_list = []
spec_average_list = []

# aupr, auc, f1, accuracy, recall, spec, precision
aupr_list = []
auc_list = []
f1_list = []
acc_list = []
precision_list = []
recall_list = []
spec_list = []


class metrics_callback(keras.callbacks.Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x_val)

        aupr, auc, f1, acc, recall, spec, precision = model_evaluate(self.y_val, y_pred)

        aupr_list.append(aupr)
        auc_list.append(auc)
        acc_list.append(acc)
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
        spec_list.append(spec)

        print('\r***auc: %s' % (str(round(auc, 4))))
        print('\r***acc: %s' % (str(round(acc, 4))))
        print('\r***f1: %s' % (str(round(f1, 4))))
        print('\r***aupr: %s' % (str(round(aupr, 4))))
        print('\r***spec: %s' % (str(round(spec, 4))))
        print('\r***recall: %s' % (str(round(recall, 4))))
        print('\r***precision: %s' % (str(round(precision, 4))))

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


# The average result of list
def average_list(list):
    average = np.array(list).mean()
    return average


# Get the results
def getResults():
    auc_average_list.append(average_list(auc_list))
    acc_average_list.append(average_list(acc_list))
    f1_average_list.append(average_list(f1_list))
    aupr_average_list.append(average_list(aupr_list))
    spec_average_list.append(average_list(spec_list))
    precision_average_list.append(average_list(precision_list))
    recall_average_list.append(average_list(recall_list))

    # After each seed calculation, empty the list
    auc_list.clear()
    acc_list.clear()
    f1_list.clear()
    aupr_list.clear()
    spec_list.clear()
    precision_list.clear()
    recall_list.clear()

    return auc_average_list, acc_average_list, f1_average_list, aupr_average_list, spec_average_list, precision_average_list, recall_average_list


# Calcaulate metrics
def get_Metrics(real_score, predict_score):
    sorted_predict_score = sorted(list(set(np.array(predict_score).flatten())))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholdlist = []
    for i in range(999):
        threshold = sorted_predict_score[int(math.ceil(sorted_predict_score_num * (i + 1) / 1000) - 1)]
        thresholdlist.append(threshold)
    thresholds = np.matrix(thresholdlist)
    TN = np.zeros((1, len(thresholdlist)))
    TP = np.zeros((1, len(thresholdlist)))
    FN = np.zeros((1, len(thresholdlist)))
    FP = np.zeros((1, len(thresholdlist)))
    for i in range(thresholds.shape[1]):
        p_index = np.where(predict_score >= thresholds[0, i])
        TP[0, i] = len(np.where(real_score[p_index] == 1)[0])
        FP[0, i] = len(np.where(real_score[p_index] == 0)[0])
        n_index = np.where(predict_score < thresholds[0, i])
        FN[0, i] = len(np.where(real_score[n_index] == 1)[0])
        TN[0, i] = len(np.where(real_score[n_index] == 0)[0])
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sen = TP / (TP + FN)
    recall = sen
    spec = TN / (TN + FP)
    precision = TP / (TP + FP)
    f1 = 2 * recall * precision / (recall + precision)
    max_index = np.argmax(f1)
    max_f1 = f1[0, max_index]
    max_accuracy = accuracy[0, max_index]
    max_recall = recall[0, max_index]
    max_spec = spec[0, max_index]
    max_precision = precision[0, max_index]
    return [max_f1, max_accuracy, max_recall, max_spec, max_precision]


# Evaluate results
def model_evaluate(real_score, predict_score):
    aupr = average_precision_score(real_score, predict_score)
    auc = roc_auc_score(real_score, predict_score)
    [f1, accuracy, recall, spec, precision] = get_Metrics(real_score, predict_score)
    return np.array([aupr, auc, f1, accuracy, recall, spec, precision])
