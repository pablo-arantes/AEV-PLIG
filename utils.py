import os
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
import torch
from sklearn.preprocessing import StandardScaler

def init_weights(layer):
    """
    function which initializes weights
    """
    if hasattr(layer, "weight") and "BatchNorm" not in str(layer):
        torch.nn.init.xavier_normal_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is True:
            torch.nn.init.zeros_(layer.bias)

class GraphDataset(InMemoryDataset):
    """
    adapted from https://github.com/thinng/GraphDTA

    class handling the dataset for a GNN
    """
    def __init__(self, root='data', dataset=None,
                 ids=None, y=None, graphs_dict=None, y_scaler=None):

        super(GraphDataset, self).__init__(root)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
            print("processed paths:")
            print(self.processed_paths, self.processed_paths[0])

        else:
            self.process(ids, y, graphs_dict)
            self.data, self.slices = torch.load(self.processed_paths[0])
            print("run through processing")
        if y_scaler is None:
            y_scaler = StandardScaler()
            y_scaler.fit(np.reshape(self.data.y, (self.__len__(),1)))
        self.y_scaler = y_scaler
        self.data.y = [torch.tensor(element[0]).float() for element in self.y_scaler.transform(np.reshape(self.data.y, (self.__len__(),1)))]

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, ids, y, graphs_dict):
        assert (len(ids) == len(y)), 'Number of datapoints and labels must be the same'
        data_list = []
        data_len = len(ids)
        for i in range(data_len):
            print('Converting unique ids to graph: {}/{}'.format(i+1, data_len))
            pdbcode = ids[i]
            label = y[i]
            c_size, features, edge_index, edge_features = graphs_dict[pdbcode]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            data_point = Data(x=torch.Tensor(np.array(features)),
                                   edge_index=torch.LongTensor(np.array(edge_index)).T,
                                   edge_attr=torch.Tensor(np.array(edge_features)),
                                   y=torch.FloatTensor(np.array([label])))
            
            # append graph, label and target sequence to data list
            data_list.append(data_point)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])
        
