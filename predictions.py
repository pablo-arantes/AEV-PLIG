import pandas as pd
import pickle
import torch
import os
from utils import GraphDataset
from torch_geometric.loader import DataLoader
from helpers import model_dict
import argparse
import numpy as np


def predict(model, device, loader, y_scaler=None):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

    return y_scaler.inverse_transform(total_labels.numpy().flatten().reshape(-1,1)).flatten(), y_scaler.inverse_transform(total_preds.detach().numpy().flatten().reshape(-1,1)).flatten()


"""
Define model_name and load scaler
"""
model_name = "20240423-200034_model_GATv2Net_pdbbind_U_bindingnet_ligsim90_"

with open('data/models/scaler.pickle','rb') as f:
    scaler = pickle.load(f)


"""
Create .pt file from graphs
"""
os.system('rm data/processed/pytorch_data.pt')

data = pd.read_csv("data/dataset_processed.csv", index_col=0)

# run below only once
with open('data/graphs.pickle', 'rb') as handle:
    graphs_dict = pickle.load(handle)

test_ids = list(data["unique_id"])
test_y = list(data["pK"])
test_data = GraphDataset(root='data', dataset='pytorch_data', ids=test_ids, y=test_y, graphs_dict=graphs_dict, y_scaler=scaler)


"""
Make predictions
"""

test_loader = DataLoader(test_data, batch_size=300, shuffle=False)

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--head', type=int, default=3)
parser.add_argument('--activation_function', type=str, default='leaky_relu')
config = parser.parse_args()

modeling = model_dict['GATv2Net']
model = modeling(node_feature_dim=test_data.num_node_features, edge_feature_dim=test_data.num_edge_features, config=config)

for i in range(10):
    model_path = 'data/models/' + model_name + str(i) + '.model'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    G_test, P_test = predict(model, torch.device('cpu'), test_loader, test_data.y_scaler)

    if(i == 0):
        df_test = pd.DataFrame(data=G_test, index=range(len(G_test)), columns=['truth'])

    col = 'preds_' + str(i)
    df_test[col] = P_test

df_test['preds'] = df_test.iloc[:,1:].mean(axis=1)

df_test['unique_id'] = data['unique_id']

data = data.merge(df_test, on='unique_id', how='left')

assert(np.allclose(data['pK'], data['truth'], rtol=1e-03))

"""
Save predictions
"""

data.to_csv("output/predictions/predictions.csv")



