from keras.layers import LSTM, Dense, Dropout, Input
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping
from keras.models import Model

import data_parameters as par
import data_cv as pre

# Get sequence max length
max_length, _ = pre.getSequenceMaxLengthAndName()

early_stopping = EarlyStopping(monitor='val_acc', patience=par.patience, verbose=2)


# Define the model
def lstm():
    input1 = Input(shape=(par.timestep, max_length * par.x_dim))
    lstm_1 = LSTM(par.lstm_1, dropout=par.dropout, return_sequences=True)(input1)
    lstm_2 = LSTM(par.lstm_2, dropout=par.dropout, return_sequences=True)(lstm_1)
    lstm_3 = LSTM(par.lstm_3, dropout=par.dropout)(lstm_2)

    dense_1 = Dense(par.dense_1, activation='relu')(lstm_3)
    dropout_1 = Dropout(par.dropout)(dense_1)
    out = Dense(par.dense_2, activation='sigmoid')(dropout_1)

    model = Model(inputs=input1, outputs=out)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Call multi gpus acceleration;  require : gpu_num >= 2
    # model = multi_gpu_model(model, gpus=par.gpu_num)

    return model
