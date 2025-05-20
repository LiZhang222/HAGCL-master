import torch
import torch.nn as nn
from dgl.ops import edge_softmax
import dgl.function as fn
from dgl.utils import expand_as_pair
from utils import create_activation, create_norm
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, from_dgl


class FALayer(MessagePassing):
    def __init__(self, num_hidden, dropout, dropmessage):
        super(FALayer, self).__init__(aggr='add')
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * num_hidden, 1)
        self.dropoutmessage = dropmessage
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def forward(self, graph, h):
        data = from_dgl(graph)
        row, col = data.edge_index

        h2 = torch.cat([h[row], h[col]], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()

        norm_degree = degree(row, num_nodes=h.size(0), dtype=torch.float).clamp(min=1)
        norm_degree = torch.pow(norm_degree, -0.5)
        norm = g * norm_degree[row] * norm_degree[col]
        norm = self.dropout(norm)
        return self.propagate(data.edge_index, size=(h.size(0), h.size(0)), x=h, norm=norm)

    def message(self, x_j, norm):
        message = norm.view(-1, 1) * x_j
        if self.training and self.dropoutmessage > 0:
            message = F.dropout(message, p=self.dropoutmessage, training=True)
        return message


class FAGCN(nn.Module):
    def __init__(self, enc_dec, num_features, num_hidden, out_dim, dropout, dropmessage, layer_num=2):
        super(FAGCN, self).__init__()
        self.layer_num = layer_num
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer(out_dim, dropout, dropmessage))
        self.t1 = nn.Linear(num_features, out_dim)
        self.t2 = nn.Linear(num_hidden, out_dim)
        self.head = nn.Identity()
        self.reset_parameters()
        self.out_dim = out_dim

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, graph, h, return_hidden=False):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        hidden_list = []
        for i in range(self.layer_num):
            h = self.layers[i](graph, h)
            h = 0.3 * raw + h
            hidden_list.append(h)

        if return_hidden:
            return self.head(h), F.normalize(self.head(h)), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)




class M_FALayer(MessagePassing):
    def __init__(self, num_hidden, heads, dropout, dropmessage, concat):
        super(M_FALayer, self).__init__(aggr='add')
        self.dropout = nn.Dropout(dropout)
        self.concat = concat
        self.dropoutmessage = dropmessage
        self.gate = nn.Linear(2 * num_hidden, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)


    def forward(self, graph, h):
        data = from_dgl(graph)
        row, col = data.edge_index
        h2 = torch.cat([h[row], h[col]], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()

        norm_degree = degree(row, num_nodes=h.size(0)).clamp(min=1)
        norm_degree = torch.pow(norm_degree, -0.5)
        norm = g * norm_degree[row] * norm_degree[col]
        norm = self.dropout(norm)
        return self.propagate(data.edge_index, size=(h.size(0), h.size(0)), x=h, norm=norm), g


    def message(self, x_j, norm):

        message = norm.view(-1, 1) * x_j
        if not self.training:
            return message
        message = F.dropout(message, self.dropoutmessage)
        return message

    def update(self, aggr_out):
        return aggr_out


class MultiHeadFALayer(nn.Module):
    def __init__(self, num_hidden, out_dim, dropout, dropmessage, heads, concat):
        super(MultiHeadFALayer, self).__init__()
        self.heads = heads
        self.concat = concat
        self.linear_cat = nn.Linear(num_hidden * heads, out_dim)
        self.linear_avg = nn.Linear(num_hidden, out_dim)
        self.head = nn.Identity()
        self.reset_parameters()
        self.attentions = nn.ModuleList([
            M_FALayer(num_hidden, heads, dropout, dropmessage, concat=concat) for _ in range(heads)])

    def reset_parameters(self):
        nn.init.xavier_normal_(self.linear_cat.weight, gain=1.414)
        nn.init.xavier_normal_(self.linear_avg.weight, gain=1.414)

    def forward(self, graph, h):
        head_outputs = []
        for att in self.attentions:
            output, g = att(graph, h)
            head_outputs.append(output)

        if self.heads == 1:
            return output
        else:
            if self.concat:
                res = torch.cat(head_outputs, dim=1)
                # return F.leaky_relu(self.linear_cat(res), 0.05)
                return torch.relu(self.linear_cat(res))
            else:
                # return F.leaky_relu(self.linear_avg(torch.mean(torch.stack(head_outputs), dim=0)), 0.05)
                return torch.relu(self.linear_avg(torch.mean(torch.stack(head_outputs), dim=0)))


class M_FAGCN(nn.Module):
    def __init__(self, num_features, num_hidden, out_dim, dropout, dropmessage, fagcn_heads, concat, layer_num=2):
        super(M_FAGCN, self).__init__()
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.out_dim = out_dim
        self.dropout = dropout
        self.layer_num = layer_num
        self.fagcn_heads = fagcn_heads
        self.concat = concat
        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(MultiHeadFALayer(num_hidden, out_dim, dropout, dropmessage, fagcn_heads, concat))
        self.t1 = nn.Linear(num_features, num_hidden)
        self.head = nn.Identity()
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)

    def forward(self, g, h, return_hidden=False):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        hidden_list = []
        for i in range(self.layer_num):
            h = self.layers[i](g, h)
            h = 0.3 * raw + h
            hidden_list.append(h)

        if return_hidden:
            return self.head(h), F.normalize(self.head(h)), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)






class GAT(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 nhead,
                 nhead_out,
                 activation,
                 feat_drop,
                 attn_drop,
                 residual,
                 norm,
                 concat_out=False,
                 encoding=False
                 ):
        super(GAT, self).__init__()
        self.out_dim = out_dim
        self.num_heads = nhead
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.concat_out = concat_out
        negative_slope = 0.2

        last_activation = create_activation(activation) if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None

        if num_layers == 1:
            self.gat_layers.append(GATConv(
                in_dim, out_dim, nhead_out,
                feat_drop, attn_drop, negative_slope, last_residual, norm=last_norm, concat_out=concat_out))
        else:
            # input projection (no residual)
            self.gat_layers.append(GATConv(
                in_dim, num_hidden, nhead,
                feat_drop, attn_drop, negative_slope, residual, create_activation(activation), norm=norm,
                concat_out=concat_out))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(GATConv(
                    num_hidden * nhead, num_hidden, nhead,
                    feat_drop, attn_drop, negative_slope, residual, create_activation(activation), norm=norm,
                    concat_out=concat_out))
            # output projection
            self.gat_layers.append(GATConv(
                num_hidden * nhead, out_dim, nhead_out,
                feat_drop, attn_drop, negative_slope, last_residual, activation=last_activation, norm=last_norm,
                concat_out=concat_out))

        # if norm is not None:
        #     self.norms = nn.ModuleList([
        #         norm(num_hidden * nhead)
        #         for _ in range(num_layers - 1)
        #     ])
        #     if self.concat_out:
        #         self.norms.append(norm(num_hidden * nhead))
        # else:
        #     self.norms = None

        self.head = nn.Identity()

    # def forward(self, g, inputs):
    #     h = inputs
    #     for l in range(self.num_layers):
    #         h = self.gat_layers[l](g, h)
    #         if l != self.num_layers - 1:
    #             h = h.flatten(1)
    #             if self.norms is not None:
    #                 h = self.norms[l](h)
    #     # output projection
    #     if self.concat_out:
    #         out = h.flatten(1)
    #         if self.norms is not None:
    #             out = self.norms[-1](out)
    #     else:
    #         out = h.mean(1)
    #     return self.head(out)

    def forward(self, g, inputs, return_hidden=False):
        h = inputs  # [2708*1433]
        # h = F.normalize(self.head(inputs), dim=1)
        hidden_list = []
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h)
            # ------------对特征进行归一化-----------------
            # h_n = F.normalize(h, dim=1)
            hidden_list.append(h)
            # h = h.flatten(1)
        # output projection
        if return_hidden:

            # return F.normalize(self.head(h), dim=1), hidden_list
            return self.head(h), F.normalize(self.head(h)), hidden_list
            # return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.num_heads * self.out_dim, num_classes)


class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=True,
                 bias=True,
                 norm=None,
                 concat_out=True):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._concat_out = concat_out

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        # if norm is not None:
        #     self.norm = norm(num_heads * out_feats)
        # else:
        #     self.norm = None

        self.norm = norm
        if norm is not None:
            self.norm = norm(num_heads * out_feats)

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise RuntimeError('There are 0-in-degree nodes in the graph, '
                                       'output for those nodes will be invalid. '
                                       'This is harmful for some applications, '
                                       'causing silent performance regression. '
                                       'Adding self-loop on the input graph by '
                                       'calling `g = dgl.add_self_loop(g)` will resolve '
                                       'the issue. Setting ``allow_zero_in_degree`` '
                                       'to be `True` when constructing this module will '
                                       'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # e[e == 0] = -1e3
            # e = graph.edata.pop('e')
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)

            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval

            if self._concat_out:
                rst = rst.flatten(1)
            else:
                rst = torch.mean(rst, dim=1)

            if self.norm is not None:
                rst = self.norm(rst)

            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst
