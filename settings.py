"""
The settings for a run.
"""

trial_name = 'Joint CNN'
log_directory = 'logs'
train_dataset_path = 'data/mini_world_expo_datasets'
validation_dataset_path = train_dataset_path
test_dataset_path = validation_dataset_path
load_model_path = None

summary_step_period = 10
number_of_epochs = 100
batch_size = 10
number_of_data_loader_workers = 0
save_epoch_period = 1000
initial_learning_rate = 1e-3
learning_rate_multiplier_function = lambda epoch: 0.1 ** (epoch / 2000)
restore_mode = 'transfer'
loss_order = 1
weight_decay = 0.1
