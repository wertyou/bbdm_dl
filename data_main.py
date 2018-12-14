import pandas as pd
import warnings
import os

import data_model
import data_cv as pre
import data_parameters as par
import data_metrics as metr
import data_util as util

aupr_average_list = []
auc_average_list = []
acc_average_list = []
f1_average_list = []
precision_average_list = []
recall_average_list = []
spec_average_list = []

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Get sequence max length
sequence_max_length, file_name = pre.getSequenceMaxLengthAndName()

# Init and visual model
model = data_model.lstm()
util.visual_model(model)

# Training process
print('*' * 20, 'Training start', '*' * 20)
for seed in range(1, par.seeds):
    print('+' * 20, 'seed:', seed, '+' * 20)
    # Clear list
    metr.clear_list()

    # Load cv data
    cv_X_train, cv_X_validation, cv_y_train, cv_y_validation, cv_train_data_num, cv_validation_data_num = pre.cv(seed)

    for i in range(par.cv):
        print('=' * 14, 'fold:', i + 1, '*' * 4, 'seed:', seed, '=' * 14)

        # Init model,clear previous weights
        model = data_model.lstm()

        # Training and validation data
        train_data = cv_X_train[i].reshape((-1, par.timestep,
                                            par.x_dim * sequence_max_length))
        validation_data = cv_X_validation[i].reshape((-1, par.timestep,
                                                      par.x_dim * sequence_max_length))

        # Training and validation label
        train_label = cv_y_train[i].reshape((-1, par.y_dim))
        validation_label = cv_y_validation[i].reshape((-1, par.y_dim))

        # Training 
        history = model.fit(train_data,
                            train_label,
                            batch_size=par.batch_size,
                            epochs=par.epochs,
                            callbacks=[data_model.early_stopping])
        # Predicting
        validation_y_pred = model.predict(validation_data, batch_size=par.batch_size)

        # Calculation metrics
        aupr, auc, f1, acc, recall, spec, precision = metr.model_evaluate(validation_label, validation_y_pred)
        metr.get_list(aupr, auc, f1, acc, recall, spec, precision)

    # Add the average results of each seed to list
    metr.get_average_results()

# Save all seeds' results to csv file
auc, acc, f1, aupr, spec, precision, recall = metr.get_results()
date_frame = pd.DataFrame(
    {'auc': auc, 'acc': acc, 'f1': f1, 'aupr': aupr, 'spec': spec, 'precision': precision, 'recall': recall})
date_frame.to_csv('./' + file_name + '_results.csv', index=True,
                  columns=['auc', 'acc', 'f1', 'aupr', 'spec', 'precision', 'recall'],
                  float_format='%.4f')

print('*' * 20, 'Training end', '*' * 20)
