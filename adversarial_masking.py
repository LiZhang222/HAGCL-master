import torch.nn as nn
from typing import Optional
from layer import GAT
import torch.nn.functional as F
from utils import create_norm


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


class AdversMask(nn.Module):
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
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            max_degree: int = 170):
        super(AdversMask, self).__init__()
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

    def forward(self, graph, x, args):
        if args.mask_encoder == 'gat':
            gnn_emb = self.mask(graph, x)
            z = F.gumbel_softmax(self.fc(gnn_emb), hard=True)
        elif args.mask_encoder == 'mlp':
            mlp_emb = self.mask(x)
            z = F.gumbel_softmax(self.fc_mlp(mlp_emb), hard=True)
        else:
            gnn_emb = self.mask(graph, x)
            z = F.gumbel_softmax(self.fc_mlp(gnn_emb), hard=True)
        return z