import argparse
import copy
import functools
import pickle

import networkx as nx
import numpy as np
import torch
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset
from torch_geometric.utils import from_networkx


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=32)


def collator(feature_name, indices):
    batch = Batch(feature_name)
    for item in indices:
        batch.append(copy.deepcopy(item))
    return batch


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class NoneScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


def generate_dataloader(train_data, eval_data, test_data, feature_name,
                        batch_size, num_workers, shuffle=True):
    train_dataset = ListDataset(train_data)
    eval_dataset = ListDataset(eval_data)
    test_dataset = ListDataset(test_data)

    collator_func = functools.partial(collator, feature_name)

    train_dataloader = DataLoaderX(dataset=train_dataset, batch_size=batch_size,
                                   num_workers=num_workers, collate_fn=collator_func,
                                   shuffle=shuffle)
    eval_dataloader = DataLoaderX(dataset=eval_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator_func,
                                  shuffle=shuffle)
    test_dataloader = DataLoaderX(dataset=test_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator_func,
                                  shuffle=False)
    return train_dataloader, eval_dataloader, test_dataloader


class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class Batch(object):

    def __init__(self, feature_name):
        self.data = {}
        self.feature_name = feature_name
        for key in feature_name:
            self.data[key] = []

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        else:
            raise KeyError('{} is not in the batch'.format(key))

    def __setitem__(self, key, value):
        if key in self.data:
            self.data[key] = value
        else:
            raise KeyError('{} is not in the batch'.format(key))

    def append(self, item):
        if len(item) != len(self.feature_name):
            raise KeyError('when append a batch, item is not equal length with feature_name')
        for i, key in enumerate(self.feature_name):
            self.data[key].append(item[i])

    def to_tensor(self, device):
        for key in self.data:
            if self.feature_name[key] == 'int':
                self.data[key] = torch.LongTensor(np.array(self.data[key])).to(device, non_blocking=True)
            elif self.feature_name[key] == 'float':
                self.data[key] = torch.FloatTensor(np.array(self.data[key])).to(device, non_blocking=True)
            else:
                raise TypeError(
                    'Batch to_tensor, only support int, float but you give {}'.format(self.feature_name[key]))

    def to_ndarray(self):
        for key in self.data:
            if self.feature_name[key] == 'int':
                self.data[key] = np.array(self.data[key])
            elif self.feature_name[key] == 'float':
                self.data[key] = np.array(self.data[key])
            else:
                raise TypeError(
                    'Batch to_ndarray, only support int, float but you give {}'.format(self.feature_name[key]))


class DataGraph():
    def __init__(self, data, lengh):
        if lengh == 519:
            graphdata = np.asarray(data[1], dtype=object)
            data_index = np.asarray(data[0], dtype=object)
        else:
            graphdata = np.asarray(data[0], dtype=object)
            data_index = np.asarray(data[1], dtype=object)
        graph = np.zeros((lengh, lengh))
        for raw, index in zip(graphdata, data_index):
            for i in raw:
                graph[index][i] = 1
        self.graph = graph


class Dataset:
    def __init__(self, args):
        self.market_name = args.market_name

        self.input_window = args.input_window
        self.output_window = args.output_window
        self.output_dim = args.output_dim
        self.timeslots = None
        self.feature_dim = None
        self.node_num = None

        self.scaler = None
        self.scaler_type = args.scaler_type

        self.batch_size = args.batch_size
        self.feature_name = {'x': 'float', 'y': 'float', 'mask': 'float'}

    def get_graph_MS(self, MS):
        graph = nx.from_numpy_array(MS)
        data = from_networkx(graph)
        edge_index = data.edge_index
        return edge_index

    def _load_dyna(self):
        if self.market_name == "NASDAQ":
            eod_data = np.load('data/NASDAQ/eod_data.npy')
            mask_data = np.load('data/NASDAQ/mask_data.npy')
            price_data = np.load('data/NASDAQ/price_data.npy')
            tickers = np.genfromtxt('data/NASDAQ/relation/NASDAQ_tickers.csv', dtype=str, skip_header=False)
            train_data = pickle.load(open('data/NASDAQ/relation/NASDAQ_File.txt', 'rb'))
            graph_data = DataGraph(train_data, len(tickers))

            pyg_graph = self.get_graph_MS(graph_data.graph)
            graph_data = graph_data.graph

        elif self.market_name == "NYSE":
            eod_data = np.load('data/NYSE/eod_data.npy')
            mask_data = np.load('data/NYSE/mask_data.npy')
            price_data = np.load('data/NYSE/price_data.npy')
            tickers = np.genfromtxt('data/NYSE/relation/NYSE_tickers.csv', dtype=str, skip_header=False)
            train_data = pickle.load(open('data/NYSE/relation/NYSE_File.txt', 'rb'))
            graph_data = DataGraph(train_data, len(tickers))

            pyg_graph = self.get_graph_MS(graph_data.graph)
            graph_data = graph_data.graph

        elif self.market_name == "snp500":
            eod_data = np.load('data/snp500/eod_data.npy')
            mask_data = np.load('data/snp500/mask_data.npy')
            price_data = np.load('data/snp500/price_data.npy')
            tickers = np.genfromtxt('data/snp500/relation/snp500_tickers.csv', dtype=str, skip_header=False)
            train_data = pickle.load(open('data/snp500/relation/snp500_File.txt', 'rb'))
            graph_data = DataGraph(train_data, len(tickers))
            pyg_graph = self.get_graph_MS(graph_data.graph)
            graph_data = graph_data.graph

        elif self.market_name == "Nikkei":
            eod_data = np.load('data/Nikkei/eod_data.npy')
            mask_data = np.load('data/Nikkei/mask_data.npy')
            price_data = np.load('data/Nikkei/price_data.npy')
            tickers = np.genfromtxt('data/Nikkei/relation/Nikkei_tickers.csv', dtype=str, skip_header=False)
            train_data = pickle.load(open('data/Nikkei/relation/Nikkei_File.txt', 'rb'))
            graph_data = DataGraph(train_data, len(tickers))
            pyg_graph = self.get_graph_MS(graph_data.graph)
            graph_data = graph_data.graph

        elif self.market_name == "SZSE":
            eod_data = np.load('data/SZSE/eod_data.npy')
            mask_data = np.load('data/SZSE/mask_data.npy')
            price_data = np.load('data/SZSE/price_data.npy')
            graph_data = np.load('data/SZSE/relation/adjSZSE.npy')
            pyg_graph = self.get_graph_MS(graph_data)
            graph_data = graph_data.graph

        elif self.market_name == "CSI500":
            eod_data = np.load('data/CSI500/eod_data.npy')
            mask_data = np.load('data/CSI500/mask_data.npy')
            price_data = np.load('data/CSI500/price_data.npy')
            tickers = np.genfromtxt('data/CSI500/relation/CSI500_tickers.csv', dtype=str, skip_header=False)
            graph_data = np.load('data/CSI500/relation/CSI500_industry_adj_matrix.npy')
            pyg_graph = self.get_graph_MS(graph_data)
        else:
            raise ValueError('Market name error!')

        correct = price_data[:, 0] == eod_data[:, 0, -1]
        print("if price data correct: ", np.all(correct))
        eod_data = eod_data.transpose(1, 0, 2)
        mask_data = mask_data.transpose(1, 0)
        self.adj = graph_data
        self.adj_index = pyg_graph
        return eod_data, mask_data, graph_data, pyg_graph

    def _generate_input_data(self, df, mask_data):
        num_samples = df.shape[0]
        x_offsets = np.sort(np.concatenate((np.arange(-self.input_window + 1, 1, 1),)))
        y_offsets = np.sort(np.arange(1, self.output_window + 1, 1))

        mask_offsets = np.concatenate((x_offsets, y_offsets))

        x, y, mask = [], [], []
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))
        for t in range(min_t, max_t):
            x_t = df[t + x_offsets, ...]
            y_t = df[t + y_offsets, ...]

            mask_t = mask_data[t + mask_offsets, ...]
            x.append(x_t)
            y.append(y_t)

            mask.append(mask_t)

        x = np.stack(x, axis=0)  # (num_samples,input_length,num_nodes,feature_dim)
        y = np.stack(y, axis=0)

        mask = np.stack(mask, axis=0)
        return x, y, mask

    def _generate_data(self):
        eod_data, mask_data, agj, adj_index = self._load_dyna()  # (T, N, C)
        self.node_num = eod_data.shape[1]
        print("EOD data shape: ", eod_data.shape)  # (T, N, C)
        print("Mask data shape: ", mask_data.shape)  # (T, N)
        x, y, mask = self._generate_input_data(eod_data, mask_data)
        y = y[:, :, :, -1]
        print("x shape: ", x.shape)  # (num_samples, input_length, num_nodes, feature_dim)
        print("y shape: ", y.shape)  # (num_samples, output_length, num_nodes)
        print("mask shape: ", mask.shape)  # (num_samples, input_length + output_length, num_nodes)

        return x, y, mask, agj, adj_index

    def _split_train_val_test(self, x, y, mask):
        num_samples = x.shape[0]
        if self.market_name == 'NASDAQ':
            num_train = 756 - 16 + 1
            num_test = 237
        elif self.market_name == 'NYSE':
            num_train = 756 - 16 + 1
            num_test = 237
        elif self.market_name == 'snp500':
            num_train = 756 - 16 + 1
            num_test = 237
        elif self.market_name == 'SZSE':
            num_train = 756 - 16 + 1
            num_test = 237
        elif self.market_name == 'Nikkei':
            num_train = 1260 - 16 + 1
            num_test = 237
        elif self.market_name == 'CSI500':
            num_train = 582 - 16 + 1
            num_test = 194
        else:
            raise ValueError('Market no split!')
        num_val = num_samples - num_test - num_train

        x_train, y_train = x[:num_train], y[:num_train]
        x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
        x_test, y_test = x[-num_test:], y[-num_test:]
        mask_train, mask_val, mask_test = mask[:num_train], mask[num_train: num_train + num_val], mask[-num_test:]

        return x_train, y_train, x_val, y_val, x_test, y_test, mask_train, mask_val, mask_test

    def _generate_train_val_test(self):
        x, y, mask, agj, adj_index = self._generate_data()
        return x, y, mask, agj, adj_index

    def _get_scalar(self, scaler_type, x_train):
        if scaler_type == "standard":
            scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
            print("mean: ", x_train.mean(), "std: ", x_train.std())
        elif scaler_type == "none":
            scaler = NoneScaler(mean=0, std=1)
        else:
            raise ValueError('Scaler type error!')
        return scaler

    def get_data(self):
        x, y, mask, agj, adj_index = self._generate_train_val_test()
        x_train, y_train, x_val, y_val, x_test, y_test, mask_train, mask_val, mask_test = self._split_train_val_test(x,
                                                                                                                     y,
                                                                                                                     mask)

        self.feature_dim = x_train.shape[-1]
        self.scaler = self._get_scalar(self.scaler_type, x_train)

        x_train = self.scaler.transform(x_train)
        y_train = self.scaler.transform(y_train)
        x_val = self.scaler.transform(x_val)
        y_val = self.scaler.transform(y_val)
        x_test = self.scaler.transform(x_test)
        y_test = self.scaler.transform(y_test)
        print("x_train shape: ", x_train.shape)
        print("y_train shape: ", y_train.shape)
        print("x_val shape: ", x_val.shape)
        print("y_val shape: ", y_val.shape)
        print("x_test shape: ", x_test.shape)
        print("y_test shape: ", y_test.shape)

        train_data = list(zip(x_train, y_train, mask_train))
        eval_data = list(zip(x_val, y_val, mask_val))
        test_data = list(zip(x_test, y_test, mask_test))
        train_dataloader, eval_dataloader, test_dataloader = generate_dataloader(
            train_data, eval_data, test_data, self.feature_name,
            self.batch_size, 0)

        return train_dataloader, eval_dataloader, test_dataloader

    def get_data_feature(self):
        print("node_num: ", self.node_num,
              "feature_dim: ", self.feature_dim,
              "output_dim: ", self.output_dim,
              "adj_index: ", self.adj_index.shape)
        return {"scaler": self.scaler,
                "num_nodes": self.node_num,
                "input_dim": self.feature_dim,
                "output_dim": self.output_dim,
                "market_name": self.market_name,
                "adj_index": self.adj_index,
                "adj": self.adj}


if __name__ == '__main__':
    args = argparse.Namespace()
    args.market_name = 'NASDAQ'
    args.input_window = 16
    args.output_window = 1
    args.output_dim = 1
    args.train_rate = 0.6
    args.valid_rate = 0.2
    args.test_rate = 0.2
    args.scaler_type = "none"
    args.batch_size = 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(args)
    train_dataloader, eval_dataloader, test_dataloader = dataset.get_data()
    data_feature = dataset.get_data_feature()
    for data in train_dataloader:
        data.to_tensor(device)
        print(data['x'].shape)
        print(data['y'].shape)
        print(data['mask'].shape)
        print(data['x'])
        print(data['y'])
        print(data['mask'])
        break
