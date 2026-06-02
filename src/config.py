vocab_size = 4096
train_split = 0.9

d_model = 128
max_seq_len = 128
num_heads = 8
dropout = 0.1

num_layers = 1
context_length = 128

batch_size = 128
learning_rate = 0.0003
epochs = 10

train_path = "data/train.npy"
val_path = "data/val.npy"


checkpoint_path = "checkpoints/epoch_2.pt"

max_new_tokens = 100
temperature = 1.0