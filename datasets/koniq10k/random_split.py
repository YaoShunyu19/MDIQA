import pickle
import random

with open('../koniq10k_official.pkl', 'rb') as f:
    data = pickle.load(f)
data_list = data[1]['train'] + data[1]['val'] + data[1]['test']

train_num = 8058

def split_data(data, seed):
    random.seed(seed)
    random.shuffle(data)
    
    train_size = 8058
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    return train_data, val_data

# 使用10个不同的seed进行划分
splits = []
for seed in range(1, 11):
    data_list = sorted(data_list)
    train_data, val_data = split_data(data_list, seed * 42)
    train_data = sorted(train_data)
    val_data = sorted(val_data)
    data = {}
    data[1] = {}
    data[1]['train'] = train_data
    data[1]['val'] = val_data
    with open(f'koniq10k_seed{seed}.pkl', 'wb') as f:
        pickle.dump(data, f)
    