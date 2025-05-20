import torch
import torch.nn as nn
from typing import Optional
from layer import GAT
import torch.nn.functional as F
from utils import create_norm
import dgl


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "mlp":
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return mod


class AdversMaskEdge(nn.Module):
    def __init__(self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            feat_drop: float,
            attn_drop: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            mask_encoder_type: str = "gcn",
            decoder_type: str = "gat",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            max_degree: int = 170):
        super(AdversMaskEdge, self).__init__()
        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        self.mask_encoder_type = mask_encoder_type
        if mask_encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1
        num_class = 2

        self.mask = setup_module(
            m_type=mask_encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation="prelu",
            dropout=feat_drop,
            attn_drop=attn_drop,
            residual=residual,
            norm=norm,
        )
        self.fc = nn.Linear(enc_num_hidden*nhead, num_class)
        self.fc_mlp = nn.Linear(enc_num_hidden, num_class)

        n_layer = num_layers * num_layers
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(num_hidden * n_layer, num_hidden))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(num_hidden, num_hidden))
        self.lins.append(torch.nn.Linear(num_hidden, num_hidden))
    #
        self.mlp = nn.Linear(num_hidden, 2)
        nn.init.xavier_normal_(self.mlp.weight, gain=1.414)
    #
    def cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer

    def cross_decoder(self, all_hidden, masked_edges, drop_out):
        dsrc = masked_edges[0]
        ddst = masked_edges[1]
        src_x = [all_hidden[i][dsrc] for i in range(len(all_hidden))]
        dst_x = [all_hidden[i][ddst] for i in range(len(all_hidden))]
        x = self.cross_layer(src_x, dst_x)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=drop_out, training=self.training)
        x = self.lins[-1](x)
        return x


    def add_hops_edge(self,graph, x, add_edge_rate, add_edges):
        if add_edge_rate == 0 or add_edges == None :
            return graph

        num_edges = graph.num_edges()
        num_nodes = graph.num_nodes()

        num_random_add_edges = int(add_edge_rate * num_edges)

        src = graph.edges()[0]
        dst = graph.edges()[1]


        (filtered_src, filtered_dst) = add_edges
        filtered_src = filtered_src.to(device=x.device)
        filtered_dst = filtered_dst.to(device=x.device)

        if len(filtered_src) > num_random_add_edges:
            rand_indices = torch.randperm(len(filtered_dst))[:num_random_add_edges]
            filtered_src = filtered_src[rand_indices]
            filtered_dst = filtered_dst[rand_indices]

        all_edge_src = torch.cat((src, filtered_src), dim=0)
        all_edge_dst = torch.cat((dst, filtered_dst), dim=0)

        add_edge_g = dgl.graph((all_edge_src, all_edge_dst), num_nodes=num_nodes)
        add_edge_g = dgl.add_self_loop(dgl.remove_self_loop(add_edge_g))
        add_edge_g.ndata['label'] = graph.ndata['label']


        return add_edge_g



    def cosine_sim(self, add_edge_graph, gnn_emb, args):
        dsrc = add_edge_graph.edges()[0]
        ddst = add_edge_graph.edges()[1]
        src_x = gnn_emb[dsrc]
        dst_x = gnn_emb[ddst]

        weight = torch.mul(src_x, dst_x)
        result = F.gumbel_softmax(self.mlp(weight), hard=True)

        return result

    def forward(self, graph, x, args, add_edges):
        if args.add_edge_rate:
            add_edge_graph = self.add_hops_edge(graph, x, args.add_edge_rate, add_edges)
        else:
            add_edge_graph = graph

        if args.mask_encoder == 'gat':
            gnn_emb, _, gnn_emb_list = self.mask(add_edge_graph, x, return_hidden=True)
            z = self.cosine_sim(add_edge_graph, gnn_emb, args)
        elif args.mask_encoder == 'mlp':
            mlp_emb = self.mask(x)
            z = self.cosine_sim(add_edge_graph, mlp_emb, args)
        else:
            gnn_emb, _, gnn_emb_list = self.mask(graph, x, return_hidden=True)
            pos_out = self.cross_decoder(gnn_emb_list, (graph.edges()[0], graph.edges()[1]), drop_out=0.5).to(torch.float32)
            z = F.gumbel_softmax(self.fc_mlp(pos_out), hard=True)
        return z, add_edge_graph