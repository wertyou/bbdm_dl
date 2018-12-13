import data_onehot
import data_parameters as par

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

'''
Goal: Load your datasets, and X is sequences, y is labels
X: your sequences
y: your labels
max_length: sequence's max length
file_name: csv results' name
'''
X, y, max_length, file_name = data_onehot.getOnehotData()


# Get sequence max length
def getSequenceMaxLengthAndName():
    return max_length, file_name


# Split train data and test data
def split_data(test_size, seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=seed,
                                                        shuffle=True)

    # Process the problem that data cannot be divisible
    X_train = X_train[:-(X_train.shape[0] - (X_train.shape[0] // par.cv) * par.cv)]
    X_test = X_test[:-(X_test.shape[0] - (X_test.shape[0] // par.cv) * par.cv)]
    y_train = y_train[:-(y_train.shape[0] - (y_train.shape[0] // par.cv) * par.cv)]
    y_test = y_test[:-(y_test.shape[0] - (y_test.shape[0] // par.cv) * par.cv)]

    return X_train, X_test, y_train, y_test


# K fold cross-validation
cv_X_train = []
cv_X_test = []
cv_y_train = []
cv_y_test = []


# cv
def cv(seed):
    # Get all train and test data
    X_train, X_test, y_train, y_test = split_data(par.test_size, seed)

    # cv split data sets
    global cv_train_data_num, cv_test_data_num
    kf = KFold(n_splits=par.cv)
    for train, test in kf.split(X_train):
        cv_X_train.append(X_train[train])
        cv_X_test.append(X_train[test])
        cv_y_train.append(y_train[train])
        cv_y_test.append(y_train[test])

        # Get the number of cross-validation training sets and test sets
        train_data_num = X_train.shape[0]
        cv_train_data_num = train_data_num // par.cv * (par.cv - 1)
        cv_test_data_num = train_data_num - cv_train_data_num

    return cv_X_train, cv_X_test, cv_y_train, cv_y_test, cv_train_data_num, cv_test_data_num
