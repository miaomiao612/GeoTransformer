#########################################################################
##   This file is part of the GeoTransformer                           ##
##                                                                     ##
##   Copyright (C) 2024 The GeoTransformer Team                        ##
##   Primary contacts: Yuhao Jia <yuhaojia98@ucla.edu>                 ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################

import argparse
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
from tqdm import tqdm
import time

parser = argparse.ArgumentParser(description='Transformer Training')

parser.add_argument('-t', '--train_type', type=str, help="Choose the para to predict, choose from 'gdp' or 'trips'", default='gdp')
parser.add_argument('-nh', '--num_heads', type=int,  help='Number of transformer attention heads', default=16)
parser.add_argument('-nl', '--num_layers', type=int, help='Number of transformer attention layers', default=4)
parser.add_argument('-k', '--knn', type=int, help='Number of querry neighbors', default=49)
parser.add_argument('-wd', '--l2_regularization', type=float, help='Size of training weight decay', default=5e-4)
parser.add_argument('-d', '--drop_out', type=float, help='Size of the tranformer drop out', default=0.05)
parser.add_argument('-bs', '--batch_size', type=int, help='Training batch size', default=4)
parser.add_argument('-lr', '--learning_rate', type=float, help='Training learning rate', default=0.01)
parser.add_argument('-e', '--epoch', type=int, help='Number of training epochs', default=60)
parser.add_argument('-wt', '--weighting_type', type=str, help="Type of the transformer weighting type, choose from 'Linear', 'IDW','Gaussian'", default='Linear')
parser.add_argument('-od', '--output_dir', type=str, help='The out_put directory of the models and the the report', default='./')

args = parser.parse_args()

this_train_type = args.train_type
this_num_heads = args.num_heads
this_num_layers = args.num_layers
this_knn = args.knn
this_weight_decay = args.l2_regularization
this_drop_out = args.drop_out
this_batch_size = args.batch_size
this_learning_rate = args.learning_rate
this_epoch = args.epoch
this_weight_type = args.weighting_type
this_output_dir = args.output_dir


# define model
class MultiHeadSpatialAttention(nn.Module):
    def __init__(self, d_model, num_heads, city_seq_length):
        super(MultiHeadSpatialAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.trainable_keys = nn.Parameter(torch.randn(city_seq_length, d_model))

    def split_heads(self, x, batch_size):
        # Change the shape of x to [batch_size, num_heads, seq_length, depth]
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_length, depth]

    def forward(self, query, values, distance_matrix, weighting):
        batch_size = query.shape[0]

        query = self.Wq(query)
        keys = self.Wk(self.trainable_keys)  # 使用可训练的 keys
        values = self.Wv(values)

        query = self.split_heads(query, batch_size)
        keys = self.split_heads(keys.unsqueeze(0).expand(batch_size, -1, -1), batch_size)
        values = self.split_heads(values, batch_size)

        # Ensure transpose is applied on the last two dimensions of keys
        keys = keys.transpose(-2, -1)  # [batch_size, num_heads, depth, n]

        scores = torch.matmul(query, keys) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))

        #1. Linear
        if weighting=='Linear':
            min_distance, _ = torch.min(distance_matrix, dim=-1, keepdim=True)
            max_distance, _ = torch.max(distance_matrix, dim=-1, keepdim=True)
            normalized_distance_matrix = (distance_matrix - min_distance) / (max_distance - min_distance + 1e-9)
            distance_weights = (1 - normalized_distance_matrix).unsqueeze(1)

        #2. Inverse Distance Weighting
        if weighting=='IDW':
            epsilon = 1e-9
            inverse_distance_matrix = 1.0 / (distance_matrix + epsilon)
            sum_inverse_distances = torch.sum(inverse_distance_matrix, dim=-1, keepdim=True)
            distance_weights = (inverse_distance_matrix / sum_inverse_distances).unsqueeze(1)

        # 3. Gaussian
        if weighting=='Gaussian':
            sigma = torch.std(distance_matrix).item()
            gaussian_weights = torch.exp(-distance_matrix ** 2 / (2 * sigma ** 2))
            sum_gaussian_weights = torch.sum(gaussian_weights, dim=-1, keepdim=True)
            distance_weights = (gaussian_weights / sum_gaussian_weights).unsqueeze(1)

        distance_weights = distance_weights.expand(-1, self.num_heads,-1, -1) 
        scores *= distance_weights.expand_as(scores)

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, values)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.num_heads * self.depth)

        return self.fc(output)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, city_seq_length, dim_feedforward=2048, dropout=0.05):
        super(TransformerDecoderLayer, self).__init__()
        self.spatial_attention = MultiHeadSpatialAttention(d_model, num_heads, city_seq_length)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            # nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, memory, distance_matrix, weighting):
        attn_output = self.spatial_attention(x, memory, distance_matrix, weighting)
        attn_output = self.dropout1(attn_output)  # Dropout is applied during training
        x = x + attn_output
        x = self.norm1(x)
        
        ff_output = self.feed_forward(x)
        ff_output = self.dropout2(ff_output)  # Dropout is applied during training
        x = x + ff_output
        x = self.norm2(x)
        
        return x

class GeoTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, city_seq_length, dim_feedforward=2048, dropout=this_drop_out):
        super(GeoTransformer, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, city_seq_length, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.output_linear = nn.Linear(d_model, 1)  

    def forward(self, x, memory, distance_matrix, weighting):
        for layer in self.layers:
            x = layer(x, memory, distance_matrix, weighting)
        x = self.output_linear(x)
        return x

def create_distance_matrix(batch_tile_names, distance_data, all_tile_names, n, all_features):
    device = all_features.device  # 获取 all_features 的设备
    batch_size = len(batch_tile_names)
    num_tiles = len(all_tile_names)
    distance_matrix = torch.full((batch_size, num_tiles), float('inf'), device=device)  # 将 distance_matrix 放在同一设备上

    for i, tile_name in enumerate(batch_tile_names):
        sanitized_tile_name = tile_name.replace('.tif', '')
        for j, other_tile_name in enumerate(all_tile_names):
            sanitized_other_tile_name = other_tile_name.replace('.tif', '')
            key_forward = f"{sanitized_tile_name}-{sanitized_other_tile_name}"
            key_reverse = f"{sanitized_other_tile_name}-{sanitized_tile_name}"
            if key_forward in distance_data:
                distance_matrix[i, j] = distance_data[key_forward]
            elif key_reverse in distance_data:
                distance_matrix[i, j] = distance_data[key_reverse]
            else:
                print(f"Missing distance data for {key_forward} and {key_reverse}")

    # 选择最近的 n 个图块
    _, indices = torch.topk(distance_matrix, n, largest=False)

    # 确保 indices 在同一设备上
    indices = indices.to(device)
    
    # 根据 indices 生成新的距离矩阵，维度为 (batch_size, 1, n)
    selected_distances = torch.gather(distance_matrix.unsqueeze(1).expand(-1, 1, -1), 2, indices.unsqueeze(1))
    
    # 生成筛选后的图块特征矩阵，维度为 (batch_size, n, feature_dim)
    selected_features = torch.gather(all_features.expand(batch_size, -1, -1), 1, indices.unsqueeze(-1).expand(-1, -1, all_features.size(-1)))

    return selected_distances, selected_features




class GeoDataset(Dataset):
    def __init__(self, features, labels, tile_names):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        self.tile_names = tile_names

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'inputs': self.features[idx],
            'labels': self.labels[idx],
            'tile_name': self.tile_names[idx]
        }
    
def calculate_r2(y_true, y_pred):
    y_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()

def filter_none(json_data):
    filtered_data = {}
    for key, value in json_data.items():
        filtered_value = {}
        for subkey, subvalue in value.items():
            if subvalue is not None:
                filtered_value[subkey] = subvalue
        if filtered_value:
            filtered_data[key] = filtered_value
    return filtered_data

# load all data

if this_train_type == 'gdp':
    # gdp
    with open('GDP_results.json', 'r') as file:
        gdp_data = json.load(file)
    gdp_list = [{'tile_name': k, 'Predictions': v['F2019GDP']} for k, v in gdp_data.items()]
    predict_df = pd.DataFrame(gdp_list)

elif this_train_type == 'trips':
    #ride-share
    with open('Trips_results.json', 'r') as file:
        gdp_data = json.load(file)
    trip_list = [{'tile_name': k, 'Predictions': v['Trips']} for k, v in gdp_data.items()]
    predict_df = pd.DataFrame(trip_list)

#1.latents
latents = pd.read_csv('latent_space.csv', index_col=0)
data = latents.join(predict_df.set_index('tile_name'))

#2.socio
# with open('ACS_results.json', 'r') as file:
#     socio_data = json.load(file)
# socio_data = filter_none(socio_data)
# df = pd.DataFrame.from_dict(socio_data, orient='index')
# data = df.join(gdp_df.set_index('tile_name'))

#3.latents+socio
# data = data.join(df)

#4.image_latents


all_tile_names = data.index.tolist()

with open('center_distances.json', 'r') as file:
    distance_data = json.load(file)

#Dataloader
batch_size = this_batch_size
features = data.drop(columns=['Predictions']).values
labels = data['Predictions'].values

random_seed = 42
torch.manual_seed(random_seed)
dataset = GeoDataset(features, labels, all_tile_names)
#train/test split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

memory_features = torch.tensor(data.drop(columns=['Predictions']).values, dtype=torch.float32)
memory = memory_features.to('cuda' if torch.cuda.is_available() else 'cpu')


# Training!

report = []

knn_city_seq = this_knn
this_weight_decay = this_weight_decay

print('Traing Parameters:')
print(f'this training epoch: {this_epoch}')
print(f'this training batch size: {this_batch_size}')
print(f'this training learning rate: {this_learning_rate}')
print(f'this training weight decay: {this_weight_decay}')
print(f'this training drop out: {this_drop_out}')
print(f'this number of transformer heads: {this_num_heads}')
print(f'this number of transformer layers: {this_num_layers}')
print(f'this weighting type of transformer: {this_weight_type}')
print(f'this knn: {knn_city_seq}')



report.append('This knn:')
report.append(knn_city_seq)
report.append('This weight_decay:')
report.append(this_weight_decay)

model = GeoTransformer(d_model=len(features[0]), num_heads=this_num_heads, num_layers=this_num_layers, city_seq_length=knn_city_seq) 
model.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=this_learning_rate, weight_decay=this_weight_decay)
curr_epoch = 0

#Loading trained model
# ckpt = torch.load('geotransformer_10.pth')
# model.load_state_dict(ckpt['model_state_dict'])
# curr_epoch = ckpt['epoch']
# optimizer.load_state_dict(ckpt['optimizer_state_dict'])
# del ckpt

# for param_group in optimizer.param_groups:
    # param_group['lr'] = 0.0001
    # param_group['weight_decay'] = 1e-3

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
num_epochs = this_epoch
alpha = 1
for epoch in range(curr_epoch, num_epochs):
    model.train()
    total_train_loss = 0
    total_train_mse = 0
    total_train_mae = 0
    
    y_true = []
    y_pred = []

    for batch_data in tqdm(train_dataloader):
        inputs = batch_data['inputs'].to(device).unsqueeze(1)
        labels = batch_data['labels'].to(device).unsqueeze(1)
        batch_tile_names = batch_data['tile_name']
        selected_distances, selected_features = create_distance_matrix(batch_tile_names, distance_data, all_tile_names, n=knn_city_seq, all_features=memory)


        selected_distances = selected_distances.to(device)
        selected_features = selected_features.to(device)
        
        optimizer.zero_grad()

        outputs = model(inputs, selected_features, selected_distances, weighting=this_weight_type)
        y_true.append(labels.cpu())
        y_pred.append(outputs.cpu())
        mse_loss = criterion_mse(outputs, labels)
        mae_loss = criterion_mae(outputs, labels)
        loss = alpha * mse_loss + (1 - alpha) * mae_loss
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        total_train_mse += mse_loss.item()
        total_train_mae += mae_loss.item()
        # print("current_loss:", loss.item(), mse_loss.item(), mae_loss.item())
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    r2_test = calculate_r2(y_true, y_pred)
    print(f'Epoch {epoch+1}, R-squared (Train): {r2_test}')
    report.append(f'Epoch {epoch+1}, R-squared (Train): {r2_test}')

    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_train_mse = total_train_mse / len(train_dataloader)
    avg_train_mae = total_train_mae / len(train_dataloader)
    print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss}, MSE Loss: {avg_train_mse}, MAE Loss: {avg_train_mae}')
    report.append(f'Epoch {epoch+1}, Train Loss: {avg_train_loss}, MSE Loss: {avg_train_mse}, MAE Loss: {avg_train_mae}')

    model.eval()
    total_test_loss = 0
    total_test_mse = 0
    total_test_mae = 0

    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_data in test_dataloader:
            inputs = batch_data['inputs'].to(device).unsqueeze(1)
            labels = batch_data['labels'].to(device).unsqueeze(1)
            batch_tile_names = batch_data['tile_name']
            selected_distances, selected_features = create_distance_matrix(batch_tile_names, distance_data, all_tile_names, n=knn_city_seq, all_features=memory)

            selected_distances = selected_distances.to(device)
            selected_features = selected_features.to(device)


            outputs = model(inputs, selected_features, selected_distances, weighting='Linear')
            y_true.append(labels.cpu())
            y_pred.append(outputs.cpu())
            mse_loss = criterion_mse(outputs, labels)
            mae_loss = criterion_mae(outputs, labels)
            loss = alpha * mse_loss + (1 - alpha) * mae_loss
            total_test_loss += loss.item()
            total_test_mse += mse_loss.item()
            total_test_mae += mae_loss.item()
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    r2_test = calculate_r2(y_true, y_pred)
    print(f'Epoch {epoch+1}, R-squared (Test): {r2_test}')
    report.append(f'Epoch {epoch+1}, R-squared (Test): {r2_test}')

    avg_test_loss = total_test_loss / len(test_dataloader)
    avg_test_mse = total_test_mse / len(test_dataloader)
    avg_test_mae = total_test_mae / len(test_dataloader)
    print(f'Epoch {epoch+1}, Average Test Loss: {avg_test_loss}, MSE Test Loss: {avg_test_mse}, MAE Test Loss: {avg_test_mae}')
    report.append(f'Epoch {epoch+1}, Average Test Loss: {avg_test_loss}, MSE Test Loss: {avg_test_mse}, MAE Test Loss: {avg_test_mae}')

    # Update scheduler with test loss
    for param_group in optimizer.param_groups:
        print(f'Epoch {epoch+1}, Current Learning Rate: {param_group["lr"]}')
    scheduler.step(avg_test_loss)

    if (epoch + 1) % 10 == 0:
        output_path = this_output_dir + f'geotransformer_{epoch+1}.pth'
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_test_loss,
        }, output_path)
        print(f"Model saved at epoch {epoch+1}")

report_path = this_output_dir + 'report.txt'
with open(report_path, 'w') as file:
    for item in report:
        file.write(f"{item}\n")