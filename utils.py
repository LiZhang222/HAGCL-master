import os
import argparse
import random
import psutil
import torch_geometric
import yaml
import logging
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from torch import optim as optim
from tensorboardX import SummaryWriter
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
import torch.nn.functional as F
from torch_sparse import SparseTensor
from sklearn.model_selection import train_test_split
import dgl
import copy
from tqdm import tqdm
# from utils import create_optimizer, accuracy
from dgl.data import (
    CoraGraphDataset,
    CiteseerGraphDataset,
    PubmedGraphDataset,
    CoraFullDataset,
    AmazonCoBuyPhotoDataset,
    CoauthorCSDataset,
    SquirrelDataset,
    ChameleonDataset,
    ActorDataset,
)
from sklearn.preprocessing import StandardScaler


GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "corafull": CoraFullDataset,
    "photo": AmazonCoBuyPhotoDataset,
    "CS": CoauthorCSDataset,
    "squirrel": SquirrelDataset,
    "chameleon": ChameleonDataset,
    "actor": ActorDataset,
}

def preprocess(graph):
    feat = graph.ndata["feat"]
    label = graph.ndata["label"]
    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]

    graph = dgl.to_bidirected(graph)

    graph.ndata["feat"] = feat
    graph.ndata["train_mask"] = train_mask
    graph.ndata["val_mask"] = val_mask
    graph.ndata["test_mask"] = test_mask
    graph.ndata["label"] = label

    # graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
    graph.create_formats_()
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


def compute_k(dataset_name, graph):
    print("adding k-hop neighbors...")

    data_dir = "hagcl/datasets/data/add"
    os.makedirs(data_dir, exist_ok=True)

    file_path = os.path.join(data_dir, f"{dataset_name}.pt")
    if os.path.exists(file_path):
        # print(f"Loading add_edges from {file_path}")
        add_edge = torch.load(file_path)
        return add_edge

    x = graph.ndata['feat']

    graph = dgl.remove_self_loop(graph)
    adj_0 = SparseTensor(row=graph.edges()[0], col=graph.edges()[1], sparse_sizes=(graph.num_nodes(), graph.num_nodes()))

    add_src = []
    add_dst = []
    adj_ = adj_0
    for i in range(6):

        adj_ = adj_0 @ adj_
        src, dst, _ = adj_.coo()
        cos_sim = compute_cosine_similarity_tensor_batched(x, adj_, 1000000)

        threshold = 0.5
        mask = cos_sim > threshold
        filtered_src = src[mask]
        filtered_dst = dst[mask]

        add_src.append(filtered_src)
        add_dst.append(filtered_dst)

    if add_src:
        add_src = torch.cat(add_src)
        add_dst = torch.cat(add_dst)
    else:
        add_src = torch.tensor([], dtype=torch.long)
        add_dst = torch.tensor([], dtype=torch.long)

    add_edge = (add_src, add_dst)

    torch.save(add_edge, file_path)
    print(f"Saved add_edges to {file_path}")

    return add_edge




def compute_cosine_similarity_tensor_batched(features, adj_, batch_size=1000):
    similarities = []
    src, dst, _ = adj_.coo()
    for i in range(0, src.shape[0], batch_size):
        src_batch = src[i:i + batch_size]
        dst_batch = dst[i:i + batch_size]
        batch_similarities = F.cosine_similarity(features[src_batch], features[dst_batch], dim=1)
        similarities.append(batch_similarities)

    return torch.cat(similarities)



def load_dataset(dataset_name):
    # assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    if dataset_name in ['chameleon', 'squirrel', 'actor', 'cora', 'citeseer', 'pubmed', 'CS', 'photo', 'corafull']:
        dataset = GRAPH_DICT[dataset_name]()


    if dataset_name in ['chameleon-filtered', 'squirrel-filtered']:
        data = np.load(os.path.join('hagcl/datasets/data', f'{dataset_name.replace("-", "_")}.npz'))
        node_features = torch.tensor(data['node_features'])
        labels = torch.tensor(data['node_labels'])
        edges = torch.tensor(data['edges'])

        graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(node_features), idtype=torch.long)

        graph.ndata['feat'] = node_features
        graph.ndata['label'] = labels

        # k sim
        add_edges = None
        # add_edges = compute_k(dataset_name, graph)

        graph = dgl.add_self_loop(dgl.remove_self_loop(graph))

        num_classes = len(labels.unique())

        train_masks = torch.tensor(data['train_masks']).permute(1, 0)
        val_masks = torch.tensor(data['val_masks']).permute(1, 0)
        test_masks = torch.tensor(data['test_masks']).permute(1, 0)

        graph.ndata['train_mask'] = train_masks
        graph.ndata['val_mask'] = val_masks
        graph.ndata['test_mask'] = test_masks
    else:
        graph = dataset[0]
        labels = graph.ndata["label"]
        num_classes = dataset.num_classes

        # k-sim
        add_edges = None
        # add_edges = compute_k(dataset_name, graph)

        graph = dgl.add_self_loop(dgl.remove_self_loop(graph))

        if dataset_name in ["photo", "CS", "computer", "physics", "corafull"]:
            # 80%test, 10%train, 10%val
            train_idx, test_idx, train_labels, test_labels = train_test_split(
                np.arange(graph.number_of_nodes()),
                labels,
                test_size=0.8)

            train_idx, val_idx, train_labels, val_labels = train_test_split(
                train_idx,
                train_labels,
                test_size=0.5)

            train_mask = torch.BoolTensor([idx in train_idx for idx in range(len(labels))])
            val_mask = torch.BoolTensor([idx in val_idx for idx in range(len(labels))])
            test_mask = torch.BoolTensor([idx in test_idx for idx in range(len(labels))])
            graph.ndata['train_mask'] = train_mask
            graph.ndata['val_mask'] = val_mask
            graph.ndata['test_mask'] = test_mask
    num_features = graph.ndata["feat"].shape[1]

    return graph, (num_features, num_classes), add_edges



def node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob=True, mute=False):
    model.eval()

    if linear_prob:
        with torch.no_grad():
            x = model.embed(graph.to(device), x.to(device))
            in_feat = x.shape[1]
        encoder = LogisticRegression(in_feat, num_classes)
    else:
        encoder = model.encoder
        encoder.reset_classifier(num_classes)

    num_finetune_params = [p.numel() for p in encoder.parameters() if p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")

    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    final_acc, estp_acc = linear_probing_for_transductive_node_classiifcation(encoder, graph, x, optimizer_f, max_epoch_f, device, mute)

    return final_acc, estp_acc


def linear_probing_for_transductive_node_classiifcation(model, graph, feat, optimizer, max_epoch, device, mute=False):
    criterion = torch.nn.CrossEntropyLoss()

    graph = graph.to(device)
    x = feat.to(device)

    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]
    labels = graph.ndata["label"]

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(graph, x)
        loss = criterion(out[train_mask], labels[train_mask])

        l2_norm_loss = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + 1e-5 * l2_norm_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(graph, x)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(
                f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(graph, x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])

    if mute:
        print(
            f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")  #

    else:
        print(
            f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    return test_acc, estp_test_acc




class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x):
        logits = self.linear(x)
        return logits








def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


def neg_sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch_geometric.seed_everything(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True)

def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    # parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--max_epoch", type=int, default=200)

    parser.add_argument("--num_heads", type=int, default=4, help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1, help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=256, help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False, help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.2, help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.1, help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--replace_rate", type=float, default=0.0)
    parser.add_argument("--encoder", type=str, default="gat")
    parser.add_argument("--mask_encoder", type=str, default="gat")
    parser.add_argument("--decoder", type=str, default="gat")
    parser.add_argument("--max_epoch_f", type=int, default=30)
    parser.add_argument("--lr_f", type=float, default=0.001, help="learning rate for evaluation")
    parser.add_argument("--weight_decay_f", type=float, default=0.0, help="weight decay for evaluation")
    parser.add_argument("--linear_prob", action="store_true", default=False)
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--concat_hidden", action="store_true", default=False)
    # Loss
    parser.add_argument("--sim", type=float, default=25.0, help='Invariance regularization loss coefficient')
    parser.add_argument("--std", type=float, default=25.0, help='Variance regularization loss coefficient')
    parser.add_argument("--cov", type=float, default=1, help='Covariance regularization loss coefficient')
    parser.add_argument("--mlp", default="256-256-256")
    parser.add_argument("--num_proj_hidden", type=int, default=512)

    parser.add_argument("--layer_dropout", type=float, default=0.)
    parser.add_argument("--drop_out", type=float, default=0.5)

    parser.add_argument("--mini_batch", type=int, default=256)
    parser.add_argument("--num_neighbor", type=int, default=[10])

    parser.add_argument("--fagcn_heads", type=int, default=1)
    parser.add_argument("--concat", type=bool, default=True)

    parser.add_argument("--belta", type=float, default=0.2)
    parser.add_argument("--belta_edge", type=float, default=0.2)
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--add_edge_rate", type=float, default=0.)
    parser.add_argument("--dropmessage", type=float, default=0.)
    parser.add_argument("--alpha_0", type=float, default=0.)
    parser.add_argument("--alpha_T", type=float, default=1.)
    parser.add_argument("--gamma", type=int, default=1)

    args = parser.parse_args()
    return args


def show_occupied_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError

    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


class TBLogger(object):
    def __init__(self, log_path="./logging_data", name="run"):
        super(TBLogger, self).__init__()

        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        self.last_step = 0
        self.log_path = log_path
        raw_name = os.path.join(log_path, name)
        name = raw_name
        for i in range(1000):
            name = raw_name + str(f"_{i}")
            if not os.path.exists(name):
                break
        self.writer = SummaryWriter(logdir=name)

    def note(self, metrics, step=None):
        if step is None:
            step = self.last_step
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        self.last_step = step

    def finish(self):
        self.writer.close()


