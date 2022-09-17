import pathlib

ABSOLUTE_PROJECT_PATH = pathlib.Path(__file__).parent.absolute()
DATA_PATH = ABSOLUTE_PROJECT_PATH / 'data'

batch_size = 32
l = 10
input_size = 1
hidden_size = 64
num_layers = 2

d = 1
len_in = 1
out_dim = 10

lr = 1e-4
weight_decay = 1e-4
epochs = 100
reg_lambda = 1e-4

log_metrics = True
experiment_name = 'lstm_ad'

evaluate_on_train_set = False
evaluate_before_training = True
eval_plots_dir = f'../saved_files/plots/{experiment_name}/'

load_saved_model = False
checkpoints_dir = f'../saved_files/checkpoints/{experiment_name}'
epoch_to_load = 99
save_model = True
epochs_saving_freq = 1
