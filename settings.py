"""
The settings for a run.
"""

trial_name = 'Test'
log_directory = '/home/golmschenk/storage/logs'
train_dataset_path = None
validation_dataset_path = '/home/golmschenk/storage/data/World Expo Datasets'
test_dataset_path = validation_dataset_path
load_model_path = None

summary_step_period = 100
number_of_epochs = 5000
batch_size = 100
number_of_data_loader_workers = 0
save_epoch_period = 5000
restore_mode = 'transfer'
loss_order = 1
weight_decay = 0.001
