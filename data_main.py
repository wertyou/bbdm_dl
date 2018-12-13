import pandas as pd
import numpy as np
import warnings
import os

import data_model
import data_cv as pre
import data_parameters as par
import data_metrics as metr

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Get sequence max length
sequence_max_length, file_name = pre.getSequenceMaxLengthAndName()

# Init model
model = data_model.lstm()

# Training process
print('*' * 20, 'Training start', '*' * 20)
for seed in range(1, par.seeds):
    print('+' * 20, 'seed:', seed, '+' * 20)

    # Load cv data
    cv_X_train, cv_X_validation, cv_y_train, cv_y_validation, cv_train_data_num, cv_validation_data_num = pre.cv(seed)

    for i in range(par.cv):
        print('=' * 14, 'fold:', i + 1, '*' * 4, 'seed:', seed, '=' * 14)

        # Training and validation data
        rna_train = cv_X_train[i].reshape((-1, par.timestep,
                                           par.x_dim * sequence_max_length))
        rna_validation = cv_X_validation[i].reshape((-1, par.timestep,
                                                     par.x_dim * sequence_max_length))
      
        # Training and validation label
        train_label = cv_y_train[i].reshape((-1, par.y_dim))
        validation_label = cv_y_validation[i].reshape((-1, par.y_dim))

        # Training
        history = model.fit(rna_train,
                            train_label,
                            batch_size=par.batch_size,
                            epochs=par.epochs,
                            callbacks=[data_model.early_stopping,
                                       metr.metrics_callback(training_data=(rna_train, train_label),
                                                             validation_data=(rna_validation,
                                                                              validation_label))])

    # Get the average results of each seed
    metr.getResults()

# Save all seeds' results to csv file
auc, acc, f1, aupr, spec, precision, recall = metr.getResults()
date_frame = pd.DataFrame(
    {'auc': auc, 'acc': acc, 'f1': f1, 'aupr': aupr, 'spec': spec, 'precision': precision, 'recall': recall})
date_frame.to_csv('./' + file_name + '_results.csv', index=True,
                  columns=['auc', 'acc', 'f1', 'aupr', 'spec', 'precision', 'recall'],
                  float_format='%.4f')

print('*' * 20, 'Training end', '*' * 20)
