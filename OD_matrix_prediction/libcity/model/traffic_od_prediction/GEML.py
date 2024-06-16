import numpy as np
import torch
import math
import torch.nn as nn

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(MultiheadAttention, self).__init__()

        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Linear layers for the query, key, and value projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # Linear layer for the output of the attention heads
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: Tensors of shape (batch_size, seq_len, d_model)
            mask: Optional mask to mask out elements in the input sequence
                  (e.g., for padding or future elements in the decoder)
        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
        """
        # Linearly project queries, keys, and values
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        # Split the queries, keys, and values into multiple heads
        q = q.view(q.size(0), -1, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), -1, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), -1, self.nhead, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        # Concatenate and linearly project the attention heads
        output = output.transpose(1, 2).contiguous().view(output.size(0), -1, self.d_model)
        output = self.out_linear(output)

        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
        Returns:
            Output tensor with positional encodings added.
        """
        batch_size, seq_len, d_model = x.size()
        x = x + self.pe.unsqueeze(0).expand(batch_size, seq_len, -1)
        return x


class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()

        self.linear1 = nn.Linear(d_model, 4*d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(4*d_model, d_model)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor after applying position-wise feedforward network.
        """
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
    
   
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Args:
            x: Input tensor
            sublayer: Sublayer module (e.g., self-attention, feedforward)
        Returns:
            Output tensor after applying layer normalization, dropout, and the sublayer.
        """
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = SublayerConnection(size, dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask to mask out elements in the input sequence
                  (e.g., for padding)
        Returns:
            Output tensor after applying self-attention and position-wise feedforward.
        """
        x = self.sublayer(x, lambda x: self.self_attn(x, x, x))
        x = self.sublayer(x, self.feed_forward)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, MultiheadAttention(d_model, nhead), PositionwiseFeedforward(d_model, dropout), dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask to mask out elements in the input sequence
                  (e.g., for padding)
        Returns:
            Output tensor after applying the specified number of encoder layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x


class SLSTM(nn.Module):
    def __init__(self, feature_dim, hidden_dim, device, p_interval):
        super(SLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell_dim = hidden_dim
        self.p_interval = p_interval
        self.f_gate = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, self.cell_dim),
            nn.Softmax(dim=1)
        )
        self.i_gate = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, self.cell_dim),
            nn.Softmax(dim=1)
        )
        self.o_gate = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, self.hidden_dim),
            nn.Softmax(dim=1)
        )
        self.g_gate = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, self.cell_dim),
            nn.Tanh()
        )
        # self.linear=nn.Linear(feature_dim*5,feature_dim)
        # self.trans=TransformerEncoder(2,feature_dim,4)

        self.tanh = nn.Tanh()

        self.device = device

    def forward(self, x):
        # x=x.permute(1,0,2)
        # x=x.reshape(x.shape[0],-1)
        # x=self.linear(x)
        # # x=self.trans(x)
        # return x

        # (T, B * N, 2E)
        h = torch.zeros((x.shape[1], self.hidden_dim)).unsqueeze(dim=0).repeat(self.p_interval, 1, 1).to(self.device)
        # (P, B * N, 2E)
        c = torch.zeros((x.shape[1], self.hidden_dim)).unsqueeze(dim=0).repeat(self.p_interval, 1, 1).to(self.device)
        # (P, B * N, 2E)

        T = x.shape[0]

        for t in range(T):
            x_ = x[t, :, :]  # (B * N, 2E)
            x_ = torch.cat((x_, h[t % self.p_interval]), 1)  # (B * N, 2E + 2E)

            f = self.f_gate(x_)  # (B * N, 2E)

            i = self.i_gate(x_)  # (B * N, 2E)

            o = self.o_gate(x_)  # (B * N, 2E)

            g = self.g_gate(x_)  # (B * N, 2E)

            c = f * c[t % self.p_interval] + i * g  # (B * N, 2E)

            c = self.tanh(c)  # (B * N, 2E)

            h[t % self.p_interval] = o * c  # (B * N, 2E)

        return h[(T - 1) % self.p_interval]  # (B * N, 2E)


class MutiLearning(nn.Module):
    def __init__(self, fea_dim, device):
        super(MutiLearning, self).__init__()
        self.fea_dim = fea_dim
        self.transition = nn.Parameter(data=torch.randn(self.fea_dim, self.fea_dim).to(device), requires_grad=True)
        self.project_in = nn.Parameter(data=torch.randn(self.fea_dim, 1).to(device), requires_grad=True)
        self.project_out = nn.Parameter(data=torch.randn(self.fea_dim, 1).to(device), requires_grad=True)

    def forward(self, x: torch.Tensor):
        # (B, N, 2E)
        x_t = x.permute(0, 2, 1)  # (B, 2E, N)

        x_in = torch.matmul(x, self.project_in)  # (B, N, 1)

        x_out = torch.matmul(x, self.project_out)  # (B, N, 1)

        x = torch.matmul(x, self.transition)
        # (B, N, 2E)
        x = torch.matmul(x, x_t)
        # (B, N, N)

        x = x.unsqueeze(dim=-1).unsqueeze(dim=1)
        x_in = x_in.unsqueeze(dim=-1).unsqueeze(dim=1)
        x_out = x_out.unsqueeze(dim=-1).unsqueeze(dim=1)

        return x, x_in, x_out


class GraphConvolution(nn.Module):
    def __init__(self, feature_dim, embed_dim, device, use_bias=False):
        super(GraphConvolution, self).__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.activation = nn.ReLU()

        # weight = torch.randn((self.feature_dim, self.embed_dim))
        weight = torch.rand((self.feature_dim, self.embed_dim))
        self.weight = nn.Parameter(data=weight.to(device), requires_grad=True)
        if use_bias:
            self.bias = nn.Parameter(data=torch.zeros(self.embed_dim).to(device), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x, a):
        # (B, N, N)
        embed = torch.matmul(a, x)
        # (B, N, N)
        embed = torch.matmul(embed, self.weight)
        # (B, N, E)
        if self.bias is not None:
            embed += self.bias

        return embed


class GCN(nn.Module):
    def __init__(self, feature_dim, embed_dim, device):
        super(GCN, self).__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.gcn = nn.ModuleList([
            GraphConvolution(feature_dim, embed_dim, device),
            GraphConvolution(embed_dim, embed_dim, device)
        ])

    def forward(self, input_seq, adj_seq):
        embed = []
        for i in range(input_seq.shape[1]):
            frame = input_seq[:, i, :, :]  # (B, N, N)
            adj = adj_seq[:, i, :, :]  # (B, N, N)

            for m in self.gcn:
                frame = m(frame, adj)
            # (B, N, E)

            embed.append(frame)

        return torch.stack(embed, dim=1)


def generate_geo_adj(distance_matrix: np.matrix):
    distance_matrix = torch.Tensor(distance_matrix)  # (N, N)
    distance_matrix = distance_matrix * distance_matrix
    sum_cost_vector = torch.sum(distance_matrix, dim=1, keepdim=True)  # (N, 1)
    weight_matrix = distance_matrix / sum_cost_vector
    weight_matrix[range(weight_matrix.shape[0]), range(weight_matrix.shape[1])] = 1
    return weight_matrix  # (N, N)


def generate_semantic_adj(demand_matrix, device):
    # (B, T, N, N)
    adj_matrix = demand_matrix.clone()#深复制
    in_matrix = adj_matrix.permute(0, 1, 3, 2)#交换最后两个维度

    adj_matrix[adj_matrix > 0] = 1

    adj_matrix[in_matrix > 0] = 1

    # 保存到文件
    # torch.save(adj_matrix, '/root/workspace/ds_od2/Bigscity-LibCity/adj_m.pth')

    degree_vector = torch.sum(adj_matrix, dim=3, keepdim=True)
    # (B, T, N, 1)
    zero_mask = (degree_vector == 0)

    # 使用 torch.nonzero 获取数值为0的位置的索引
    zero_indices = torch.nonzero(zero_mask, as_tuple=False)

    sum_degree_vector = torch.matmul(adj_matrix, degree_vector)
    # 防止度和向量中的0值，避免后续的除零问题
    # (B, T, N, 1)


    weight_matrix = torch.matmul(1 / (sum_degree_vector+1e-3), degree_vector.permute((0, 1, 3, 2)))  # (B, T, N, N)

    weight_matrix[:, :, range(weight_matrix.shape[2]), range(weight_matrix.shape[3])] = 1
    for indices in zero_indices:
        weight_matrix[indices[0], indices[1], indices[2], :] = 0

    return weight_matrix


class GEML(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.num_nodes = self.data_feature.get('num_nodes')
        self._scaler = self.data_feature.get('scaler')
        self.output_dim = config.get('output_dim')
        self.device = config.get('device', torch.device('cpu'))
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)

        self.p_interval = config.get('p_interval', 1)
        self.embed_dim = config.get('embed_dim')
        self.batch_size = config.get('batch_size')
        self.loss_p0 = config.get('loss_p0', 0.5)
        self.loss_p1 = config.get('loss_p1', 0.25)
        self.loss_p2 = config.get('loss_p2', 0.25)

        dis_mx = self.data_feature.get('adj_mx')
        self.geo_adj = generate_geo_adj(dis_mx) \
            .repeat(self.batch_size * self.input_window, 1) \
            .reshape((self.batch_size, self.input_window, self.num_nodes, self.num_nodes)) \
            .to(self.device)
        
        # print(f"The max value of self.geo_adj is {torch.max(self.geo_adj)}")
        # print(f"The min value of self.geo_adj is {torch.min(self.geo_adj)}")
        # print(f"The mean value of self.geo_adj is {torch.mean(self.geo_adj)}")
       

        self.GCN_ge = GCN(self.num_nodes, self.embed_dim, self.device)
        self.GCN_se = GCN(self.num_nodes, self.embed_dim, self.device)

        # self.LSTM = nn.LSTM(2 * self.embed_dim, 2 * self.embed_dim)
        self.LSTM = SLSTM(2 * self.embed_dim, 2 * self.embed_dim, self.device, self.p_interval)

        self.mutiLearning = MutiLearning(2 * self.embed_dim, self.device)

    def forward(self, batch):
        x = batch['X'].squeeze(dim=-1)
        # print(f"The max value of x is {torch.max(x)}")
        # (B, T, N, N)
        x_ge_embed = self.GCN_ge(x, self.geo_adj[:x.shape[0], ...])
        # x_ge_embed=(x_ge_embed-torch.mean(x_ge_embed))/torch.std(x_ge_embed)
        # (B, T, N, E)
        # print(f"The max value of x_ge_embed is {torch.max(x_ge_embed)}")
        # print(f"The min value of x_ge_embed is {torch.min(x_ge_embed)}")
        # print(f"The mean value of x_ge_embed is {torch.mean(x_ge_embed)}")

        x_se_embed = self.GCN_se(x, self.semantic_adj)
        # x_se_embed=(x_se_embed-torch.mean(x_se_embed))/torch.std(x_se_embed)
        # print(f"The max value of x_se_embed is {torch.max(x_se_embed)}")
        # print(f"The min value of x_ge_embed is {torch.min(x_ge_embed)}")
        # print(f"The mean value of x_ge_embed is {torch.mean(x_ge_embed)}")

        

        # (B, T, N, E)
        x_embed = torch.cat([x_ge_embed, x_se_embed], dim=3)
        # (B, T, N, 2E)
        x_embed = x_embed.permute(1, 0, 2, 3)
        # (T, B, N, 2E)
        x_embed = x_embed.reshape((self.input_window, -1, 2 * self.embed_dim))
        # (T, B * N, 2E)

        # _, (h, _) = self.LSTM(x_embed)
        # x_embed_pred = h[0].reshape((self.batch_size, -1, 2 * self.embed_dim))
        # print(f"The shape of x_embed_pred is {x_embed.shape}")
        # print(f"The max value of x_embed is {torch.max(x_embed)}")
        x_embed_pred = self.LSTM(x_embed).reshape((x.shape[0], -1, 2 * self.embed_dim))
        # (B, N, 2E)

        out = self.mutiLearning(x_embed_pred)

        return out

    def calculate_loss(self, batch):
        y_true = batch['y']  # (B, TO, N, N, 1)
        y_in_true = torch.sum(y_true, dim=-2, keepdim=True)  # (B, TO, N, 1)
        y_out_true = torch.sum(y_true.permute(0, 1, 3, 2, 4), dim=-2, keepdim=True)  # (B, TO, N, 1)
        y_pred, y_in, y_out = self.predict(batch)

        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_in_true = self._scaler.inverse_transform(y_in_true[..., :self.output_dim])
        y_out_true = self._scaler.inverse_transform(y_out_true[..., :self.output_dim])

        y_pred = self._scaler.inverse_transform(y_pred[..., :self.output_dim])
        y_in = self._scaler.inverse_transform(y_in[..., :self.output_dim])
        y_out = self._scaler.inverse_transform(y_out[..., :self.output_dim])

        loss_pred = loss.masked_mse_torch(y_pred, y_true)
        loss_in = loss.masked_mse_torch(y_in, y_in_true)
        loss_out = loss.masked_mse_torch(y_out, y_out_true)
        return self.loss_p0 * loss_pred + self.loss_p1 * loss_in + self.loss_p2 * loss_out

    def predict(self, batch):
        x = batch['X']  # (B, T, N, N, 1)
        self.semantic_adj = generate_semantic_adj(x.squeeze(dim=-1), self.device)
        # print(f"The max value of semantic_adj is {torch.max(self.semantic_adj)}")
        # print(f"The min value of semantic_adj is {torch.min(self.semantic_adj)}")
        # print(f"The mean value of semantic_adj is {torch.mean(self.semantic_adj)}")
        assert x.shape[-1] == 1 or print("The feature_dim must be 1")
        y_pred = []
        y_in_pred = []
        y_out_pred = []
        x_ = x.clone()
        for i in range(self.output_window):
            batch_tmp = {'X': x_}
            y_, y_in_, y_out_ = self.forward(batch_tmp)  # (B, 1, N, N, 1)
            y_pred.append(y_.clone())
            y_in_pred.append(y_in_.clone())
            y_out_pred.append(y_out_.clone())

            x_ = torch.cat([x_[:, 1:, :, :, :], y_], dim=1)

        y_pred = torch.cat(y_pred, dim=1)  # (B, TO, N, N, 1)
        y_in_pred = torch.cat(y_in_pred, dim=1)  # (B, TO, N, 1)
        y_out_pred = torch.cat(y_out_pred, dim=1)  # (B, TO, N, 1)
        return y_pred, y_in_pred, y_out_pred
