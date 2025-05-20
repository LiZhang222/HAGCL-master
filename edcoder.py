from typing import Optional
from itertools import chain
from functools import partial
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from layer import FAGCN
from layer import M_FAGCN
from layer import GAT
from utils import create_norm, sce_loss, neg_sce_loss
import dgl
from dgl import add_self_loop, remove_self_loop


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, in_drop, activation, residual, norm, nhead,
                 nhead_out, attn_drop, drop_out, dropmessage, fagcn_heads, concat, concat_out=True) -> nn.Module:
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
            feat_drop=in_drop,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=in_drop,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=in_drop,
            activation=activation,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=in_drop,
            activation=activation,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "sage":
        mod = SAGE(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=in_drop,
            activation=activation,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "fagcn":
        mod = FAGCN(
            enc_dec=enc_dec,
            num_features=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            dropout=drop_out,
            dropmessage=dropmessage,
            layer_num=num_layers,
        )
    elif m_type == "m_fagcn":
        mod = M_FAGCN(
            num_features=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            dropout=drop_out,
            dropmessage=dropmessage,
            fagcn_heads=fagcn_heads,
            layer_num=num_layers,
            concat=concat,
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


class PreModel(nn.Module):
    def __init__(
            self,
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
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            drop_out: float = 0.5,
            mlp: str = "512-512-512",
            num_proj_hidden: int = 512,
            layer_dropout: float = 0.,
            fagcn_heads: int = 1,
            concat: bool = True,
            add_edge_rate: float = 0.2,
            dropmessage: float = 0.
    ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden

        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self._drop_out = drop_out
        self.feature_dim = in_dim
        self.mlp = mlp
        self.num_proj_hidden = num_proj_hidden
        self.layer_dropout = layer_dropout
        self.fagcn_heads = fagcn_heads
        self.concat = concat
        self.add_edge_rate = add_edge_rate
        self.dropmessage = dropmessage

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden

        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation='prelu',
            in_drop=feat_drop,
            attn_drop=attn_drop,
            residual=residual,
            norm=norm,
            drop_out=drop_out,
            dropmessage=dropmessage,
            fagcn_heads=fagcn_heads,
            concat=concat
        )

        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation="prelu",
            in_drop=feat_drop,
            attn_drop=attn_drop,
            residual=residual,
            norm=norm,
            concat_out=True,
            drop_out=drop_out,
            dropmessage=dropmessage,
            fagcn_heads=fagcn_heads,
            concat=concat
        )

        n_layer = num_layers * num_layers
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(num_hidden * n_layer, num_hidden))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(num_hidden, num_hidden))
        self.lins.append(torch.nn.Linear(num_hidden, 1))

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
            self.projector = Projector(self.mlp, num_hidden * num_layers, layer_dropout)
            self.fc1 = torch.nn.Linear(num_hidden * num_layers, num_proj_hidden)
            self.fc2 = torch.nn.Linear(num_proj_hidden * num_layers, num_hidden)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
            self.projector = Projector(self.mlp, num_hidden, layer_dropout)
            self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
            self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)


        nn.init.xavier_normal_(self.lins[0].weight, gain=1.414)
        for layer_ in range(num_layers - 2):
            layer = self.lins[layer_]
            nn.init.xavier_normal_(layer.weight, gain=1.414)
        nn.init.xavier_normal_(self.lins[-1].weight, gain=1.414)

        nn.init.xavier_normal_(self.encoder_to_decoder.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)


        self.criterion, self.mask_criterion = self.setup_loss_fn("sce", alpha_l)


    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
            mask_criterion = -nn.MSELoss()

        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
            mask_criterion = partial(neg_sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion, mask_criterion


    def encoding_mask_noise(self, g, x, args, mask_rate, mask_prob, epoch):

        num_nodes = g.num_nodes()
        alpha_adv = args.alpha_0 + ((epoch / args.max_epoch) ** args.gamma) * (
                    args.alpha_T - args.alpha_0)

        # random masking
        perm = torch.randperm(num_nodes, device=x.device)
        num_random_mask_nodes = int(mask_rate * num_nodes * (1. - alpha_adv))
        random_mask_nodes = perm[: num_random_mask_nodes]
        random_keep_nodes = perm[num_random_mask_nodes:]

        # adversarial masking
        mask_ = mask_prob[:, 1]
        perm_adv = torch.randperm(num_nodes, device=x.device)
        adv_keep_token = perm_adv[:int(num_nodes * (1. - alpha_adv))]
        mask_[adv_keep_token] = 1.
        Mask_ = mask_.reshape(-1, 1)

        adv_keep_nodes = mask_.nonzero().reshape(-1)
        adv_mask_nodes = (1 - mask_).nonzero().reshape(-1)

        mask_nodes = torch.cat((random_mask_nodes, adv_mask_nodes), dim=0).unique()
        keep_nodes = torch.tensor(np.intersect1d(random_keep_nodes.cpu().numpy(), adv_keep_nodes.cpu().numpy())).to(
            x.device)

        num_mask_nodes = mask_nodes.shape[0]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            if int(self._replace_rate * num_mask_nodes) == 0:
                noise_nodes = mask_nodes[perm_mask[num_mask_nodes:]]
            else:
                noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x = out_x * Mask_
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            out_x = out_x * Mask_
            token_nodes = mask_nodes
            out_x[token_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes), alpha_adv

    def mask_edge(self, graph, x, args, mask_rate, mask_prob_edge, epoch):

        num_edges = graph.num_edges()
        alpha_adv_edge = args.alpha_0 + ((epoch / args.max_epoch) ** args.gamma) * (args.alpha_T - args.alpha_0)

        src = graph.edges()[0]
        dst = graph.edges()[1]

        # random masking
        perm = torch.randperm(num_edges, device=x.device)
        num_random_mask_edges = int(mask_rate * num_edges * (1. - alpha_adv_edge))
        random_mask_edges = perm[: num_random_mask_edges]
        random_keep_edges = perm[num_random_mask_edges:]

        # adversarial masking
        mask_ = mask_prob_edge[:, 1]
        perm_adv = torch.randperm(num_edges, device=x.device)
        adv_keep_token = perm_adv[:int(num_edges * (1. - alpha_adv_edge))]
        mask_[adv_keep_token] = 1.
        Mask_ = mask_.reshape(-1, 1)

        adv_keep_edges = mask_.nonzero().reshape(-1)
        adv_mask_edges = (1 - mask_).nonzero().reshape(-1)

        mask_edges = torch.cat((random_mask_edges, adv_mask_edges), dim=0).unique()  # 6195
        keep_edges = torch.tensor(np.intersect1d(random_keep_edges.cpu().numpy(), adv_keep_edges.cpu().numpy())).to(x.device)  # 7069


        mask_edges_src = src[mask_edges]
        mask_edges_dst = dst[mask_edges]
        keep_edges_src = src[keep_edges]
        keep_edges_dst = dst[keep_edges]

        ng = dgl.graph((keep_edges_src, keep_edges_dst), num_nodes=graph.num_nodes())
        ng = add_self_loop(ng)

        return ng, (mask_edges_src, mask_edges_dst), alpha_adv_edge

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
        return torch.sigmoid(x)

    def mask_attr_prediction(self, g, add_edge_graph, x, epoch, args, mask_prob, mask_prob_edge, pooler):

        pre_use_g, use_x, (mask_nodes, keep_nodes), alpha_adv = self.encoding_mask_noise(g, x, args, self._mask_rate, mask_prob, epoch)
        use_g, masked_edges, alpha_adv_edge = self.mask_edge(add_edge_graph, x, args, self._drop_edge_rate, mask_prob_edge, epoch)

        enc_rep_node, enc_rep_node_n, all_hidden_node = self.encoder(pre_use_g, use_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep_node = torch.cat(all_hidden_node, dim=1)
        rep_node = self.encoder_to_decoder(enc_rep_node)

        enc_rep_edge, enc_rep_edge_n, all_hidden_edge = self.encoder(use_g, x, return_hidden=True)
        if self._concat_hidden:
            enc_rep_edge = torch.cat(all_hidden_edge, dim=1)
        rep_edge = self.encoder_to_decoder(enc_rep_edge)

        src_x = enc_rep_edge[masked_edges[0]]
        dst_x = enc_rep_edge[masked_edges[1]]
        cos_sim = F.cosine_similarity(src_x, dst_x, dim=1)
        pos_loss = -(cos_sim + 1e-15).mean()
        edge_loss = pos_loss


        if self._decoder_type not in ("mlp", "linear"):
            rep_node[mask_nodes] = 0.0
        if self._decoder_type in ("mlp", "linear"):
            recon = self.decoder(rep_node)
        else:
            recon = self.decoder(pre_use_g, rep_node)


        x_init_mask = x[mask_nodes]
        x_rec_mask = recon[mask_nodes]  # #
        node_loss = self.criterion(x_rec_mask, x_init_mask)


        num_all_noeds = mask_prob[:, 1].sum() + mask_prob[:, 0].sum()
        loss_mask = -self.mask_criterion(x_rec_mask, x_init_mask) + args.belta * (
                torch.tensor([1.]).to(g.device) / torch.sin(torch.pi / num_all_noeds * (mask_prob[:, 0].sum())))


        num_all_edges = mask_prob_edge[:, 1].sum() + mask_prob_edge[:, 0].sum()
        loss_mask_edge = -(edge_loss) + args.belta_edge * (
                torch.tensor([1.]).to(g.device) / torch.sin(torch.pi / num_all_edges * (mask_prob_edge[:, 0].sum())))

        # VICReg Loss
        h1 = self.projection(enc_rep_node, args.layer_dropout)
        h2 = self.projection(enc_rep_edge, args.layer_dropout)

        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        ret = (l1 + l2) * 0.5
        sim_loss = ret.mean()

        edge = self.projector(enc_rep_edge)
        node = self.projector(enc_rep_node)

        xx = edge - edge.mean(dim=0)
        y = node - node.mean(dim=0)

        std_x = torch.sqrt(xx.var(dim=0) + 1e-15)
        std_y = torch.sqrt(y.var(dim=0) + 1e-15)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2  # Variance

        batch_size = enc_rep_edge_n.size(0)
        feature_dim = enc_rep_edge_n.size(1)

        lambda_ = 1. / feature_dim

        z1_norm = (enc_rep_edge_n - enc_rep_edge_n.mean(dim=0)) / (enc_rep_edge_n.std(dim=0) + 1e-15)
        z2_norm = (enc_rep_node_n - enc_rep_node_n.mean(dim=0)) / (enc_rep_node_n.std(dim=0) + 1e-15)
        c = (z1_norm.T @ z2_norm) / batch_size

        off_diagonal_mask = ~torch.eye(feature_dim).bool()
        cov_loss = (1 - c.diagonal()).pow(2).sum().div(node.shape[1])
        cov_loss += lambda_ * c[off_diagonal_mask].pow(2).sum().div(node.shape[1])


        loss = args.std * std_loss + args.sim * sim_loss + args.cov * cov_loss

        return loss, loss_mask, loss_mask_edge

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    def forward(self, g, add_edge_graph, x, epoch, args, mask_prob, mask_prob_edge, pooler):
        loss, loss_mask, loss_mask_edge = self.mask_attr_prediction(g, add_edge_graph, x, epoch, args, mask_prob, mask_prob_edge, pooler)
        loss_item = {"loss": loss.item()}
        return loss, loss_mask, loss_mask_edge, loss_item

    def projection(self, z: torch.Tensor, layer_dropout) -> torch.Tensor:
        z = F.dropout(z, p=layer_dropout, training=self.training)
        z = F.elu(self.fc1(z))
        z = F.dropout(z, p=layer_dropout, training=self.training)
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / 0.4)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))



    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def Projector(mlp, embedding, layer_dropout):
    mlp_spec = f"{embedding}-{mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        linear_layer = nn.Linear(f[i], f[i + 1])
        nn.init.xavier_normal_(linear_layer.weight, gain=1.414)

        layers.append(linear_layer)
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.Dropout(p=layer_dropout))
        layers.append(nn.PReLU())

    last_linear_layer = nn.Linear(f[-2], f[-1], bias=False)
    nn.init.xavier_normal_(last_linear_layer.weight, gain=1.414)
    layers.append(last_linear_layer)
    layers.append(nn.Dropout(p=layer_dropout))
    return nn.Sequential(*layers)






