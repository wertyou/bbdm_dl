# Define the parameters

'''
The parameters of preprocessing file
'''

# Training set and test set ratio
test_size = 0.2
# Random seed
random_state = 0
# Set seeds
seeds = 3
# CV
cv = 3


'''
The parameters of model file
'''

# dropout
dropout = 0.5
# timestep
timestep = 1
# dim
x_dim = 5
y_dim = 2
# batch_size
batch_size = 128
# epoch
epochs = 3
# early_stopping patience
patience = 5
# The number of lstm_1 hidden layer nodes
lstm_1 = 512
# The number of lstm_2 hidden layer nodes
lstm_2 = 256
# The number of lstm_2 hidden layer nodes
lstm_3 = 128
# The number of dense_1 hidden layer nodes
dense_1 = 64
# The number of dense_2 hidden layer nodes
dense_2 = 2

'''
Others parameters 
'''
# The gpu numbers
gpu_num = 1