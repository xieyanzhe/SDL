import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AntiSymmetricConv


class AntiSymmetric(torch.nn.Module):
    def __init__(self, num_node_features):
        super(AntiSymmetric, self).__init__()
        self.conv1 = AntiSymmetricConv(num_node_features)
        self.conv2 = AntiSymmetricConv(num_node_features)

    def forward(self, data, edge_index):
        x, edge_index = data, edge_index
        x_1 = self.conv1(x, edge_index)
        x_1 = F.leaky_relu(x_1)
        x_2 = self.conv2(x + x_1, edge_index)
        x = F.softmax(x_2, dim=-1)
        return x


class TemporalEncoder(nn.Module):
    def __init__(self, num_stocks, encoder_dim, temporal_embed_dim, input_time):
        super(TemporalEncoder, self).__init__()
        self.input_time = input_time
        self.stock_embedding = nn.Embedding(num_stocks, temporal_embed_dim)
        self.stock_encoder = nn.Linear(input_time * 9, encoder_dim)
        self.embed_encoder = nn.Linear(temporal_embed_dim, encoder_dim)
        self.temporal_embed_dim = temporal_embed_dim
        self.encoder_dim = encoder_dim
        self.dropout = nn.Dropout(0.1)

    def forward(self, stock):
        batch_size, time_len, num_stocks, features = stock.shape

        stock_indices = torch.arange(num_stocks).to(stock.device)
        stock_indices = stock_indices.unsqueeze(0).expand(batch_size, num_stocks)

        stock_embeds = self.stock_embedding(stock_indices)
        stock_embeds = self.dropout(stock_embeds)

        stock = stock.permute(0, 2, 1, 3).reshape(batch_size, num_stocks, time_len * features)
        stock = self.stock_encoder(stock)

        stock_features = self.embed_encoder(stock_embeds)

        p = F.sigmoid(stock_features)
        q = F.tanh(stock)

        output = p * q + (1 - p) * stock_features  # [B, N, output_dim]
        output = output.view(batch_size, num_stocks, self.encoder_dim)  # [B, N, output_dim]

        return output


class GTUnit(nn.Module):
    def __init__(self, input_dim):
        super(GTUnit, self).__init__()
        self.input_dim = input_dim
        self.gate = nn.Linear(input_dim, input_dim)
        self.update = nn.Linear(input_dim, input_dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        p = self.sigmoid(self.gate(x))
        q = self.tanh(self.update(x))
        h = p * q + (1 - p) * x
        return h


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gtu = GTUnit(input_dim)
        self.fc = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, stock):
        stock = self.gtu(stock)
        stock = self.fc(stock)
        return stock


class DyGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheby_k, embed_dim, node_num, aggregate_type='sum'):
        super(DyGCN, self).__init__()
        self.cheby_k = cheby_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheby_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        if aggregate_type == 'weighted_sum':
            self.weights_cheby = torch.nn.Parameter(torch.ones(cheby_k))
        self.aggregate_type = aggregate_type
        self.node_num = node_num
        self.mask = torch.zeros(node_num, node_num)

    def forward(self, x, all_emb, stock_emb, return_supports=False):
        batch_size, node_num, _ = all_emb.shape
        A = F.relu(torch.matmul(all_emb, all_emb.transpose(1, 2)))  # [B, N, N]
        supports = F.softmax(A, dim=-1)  # [B, N, N]

        t_k_0 = torch.eye(node_num).to(supports.device)  # [B, N, N]
        t_k_0 = t_k_0.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, N]
        support_set = [t_k_0, supports]
        for k in range(2, self.cheby_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports_cheby = torch.stack(support_set, dim=0)  # [cheby_k, B, N, N]
        supports_cheby = supports_cheby.permute(1, 0, 2, 3)  # [B, cheby_k, N, N]

        # B, N, cheby_k, dim_in, dim_out
        weights = torch.einsum('bni,ikop->bnkop', stock_emb, self.weights_pool)
        # B, N, dim_out
        bias = torch.matmul(stock_emb, self.bias_pool)
        # B, cheby_k, N, dim_in
        x_g = torch.einsum('bkij,bjd->bkid', supports_cheby, x)
        # B, N, cheby_k, dim_out
        x_g_conv = torch.einsum('bkni,bnkio->bnko', x_g, weights)
        # B, N, dim_out
        if self.aggregate_type == 'sum':
            x_g_conv = x_g_conv.sum(dim=2) + bias
        elif self.aggregate_type == 'weighted_sum':
            x_g_conv = x_g_conv * self.weights_cheby.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            x_g_conv = x_g_conv.sum(dim=2) + bias

        if return_supports:
            return x_g_conv, supports
        return x_g_conv


class SDL(nn.Module):
    def __init__(self, config, data_feature):
        super().__init__()
        self.input_dim = data_feature['input_dim']
        self.output_dim = data_feature.get('output_dim')
        self.input_time = config['input_time']
        self.output_time = config['output_time']
        self.node_num = data_feature['num_nodes']
        self.scaler = data_feature['scaler']
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.adj_static = torch.tensor(data_feature['adj'], dtype=torch.float32).to(self.device)
        self.adj_index_static = data_feature['adj_index'].to(self.device)

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.gcn_dim = config['gcn_dim']
        self.encoder_dim = config['encoder_dim']
        self.temporal_embed_dim = config['temporal_embed_dim']

        self.cheby_k = config['cheby_k']

        self.temporal_encoder = TemporalEncoder(num_stocks=self.node_num, encoder_dim=self.encoder_dim,
                                                temporal_embed_dim=self.temporal_embed_dim,
                                                input_time=self.input_time)
        self.dygcn = DyGCN(dim_in=self.input_time * self.input_dim, dim_out=self.gcn_dim, cheby_k=self.cheby_k,
                           embed_dim=self.encoder_dim, node_num=self.node_num)
        self.decoder = Decoder(input_dim=self.gcn_dim, output_dim=self.output_dim * self.output_time)
        self.AntisSymmetric = AntiSymmetric(self.node_num)

        self.gcn_activation = nn.GELU()
        self.temporal_norm = nn.LayerNorm(self.encoder_dim)

        self._init_parameters()

    def forward(self, data):
        # x [B, T, N, C]
        x = data['x']
        stock = x  # [B,T,N,2]

        temporal_stock = self.temporal_encoder(stock)  # [B,N,64]
        temporal_stock = self.temporal_norm(temporal_stock)

        stock = stock.permute(0, 2, 1, 3)  # [B,N,T,C]
        stock = stock.reshape(stock.shape[0], stock.shape[1], -1)  # [B,N,T*C]

        gcn_output, adj_dy = self.dygcn(stock, temporal_stock, temporal_stock, return_supports=True)  # [B,N,64]
        gcn_output = self.gcn_activation(gcn_output)  # [B,N,64]

        output = self.decoder(gcn_output)  # [B,N,4*2]
        output = output.reshape(output.shape[0], output.shape[1], self.output_time, self.output_dim)  # [B,N,T,C]
        output = output.permute(0, 2, 1, 3)  # [B,T,N,C]

        return output, adj_dy

    def predict(self, data):
        with torch.no_grad():
            return self.forward(data)[0]

    def _init_parameters(self):
        print('Initializing parameters...')
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def cal_train_loss(self, data):
        pred, adj_dy = self.forward(data)

        noise = torch.normal(0, 0.01, (self.node_num, self.node_num)).to(self.device)
        adj_static_noise = self.adj_static + noise
        adj_anti = self.AntisSymmetric(adj_static_noise, self.adj_index_static)
        loss_graph = torch.norm(adj_anti - adj_dy, p='fro') / torch.norm(adj_anti, p='fro')

        pred = self.scaler.inverse_transform(pred).squeeze(-1)
        truth = self.scaler.inverse_transform(data['y'])
        data['x'] = self.scaler.inverse_transform(data['x'])

        mask = data['mask'].min(dim=1)[0].unsqueeze(1)

        mask /= torch.mean(mask)
        huber_loss = F.huber_loss(pred, truth, reduction='none')
        huber_loss.mul_(mask)
        huber_loss = torch.mean(huber_loss)

        rank_loss = []
        for i in range(pred.shape[0]):
            cur_mask = mask[i].view(-1).bool()
            cur_pred = pred[i].view(-1, 1)[cur_mask]
            cur_truth = truth[i].view(-1, 1)[cur_mask]
            last_price = data['x'][i, -1, :, -1].view(-1, 1)[cur_mask]
            return_ratio = torch.div(torch.sub(cur_pred, last_price), last_price)
            truth_ratio = torch.div(torch.sub(cur_truth, last_price), last_price)
            all_one = torch.ones(cur_pred.shape[0], 1, dtype=torch.float32).to(self.device)
            pre_pw_dif = torch.sub(return_ratio @ all_one.t(), all_one @ return_ratio.t())
            gt_pw_dif = torch.sub(all_one @ truth_ratio.t(), truth_ratio @ all_one.t())
            rank_loss.append(torch.mean(F.relu(pre_pw_dif * gt_pw_dif)))
        rank_loss = torch.mean(torch.stack(rank_loss))

        alpha = 10
        return huber_loss + alpha * rank_loss + loss_graph * 1
