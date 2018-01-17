"""
The settings for a run.
"""

trial_name = 'base'
log_directory = '../logs'
train_dataset_path = '../data'
validation_dataset_path = train_dataset_path
test_dataset_path = validation_dataset_path
load_model_path = None

summary_step_period = 100
number_of_epochs = 300000
batch_size = 400
number_of_data_loader_workers = 4
save_epoch_period = 2500
restore_mode = 'transfer'
loss_order = 1
weight_decay = 0.01
