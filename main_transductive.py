import logging
import numpy as np
from tqdm import tqdm
import torch

from utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    load_best_configs,
    load_dataset,
    node_classification_evaluation
)
from model import build_model

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def pretrain(model, mask_module_edge, optimizer_mask_edge, graph, feat, optimizer, max_epoch, device, scheduler, args, mask_module, optimizer_mask, add_edges):
    graph = graph.to(device)
    x = feat.to(device)

    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()
        mask_module.train()
        mask_module_edge.train()

        # optimize mask_module and mask_module_edge
        mask_prob = mask_module.forward(graph, x, args)
        mask_prob_edge, add_edge_graph = mask_module_edge.forward(graph, x, args, add_edges)

        loss, loss_mask, loss_mask_edge, loss_dict = model(graph, add_edge_graph, x, epoch, args, mask_prob, mask_prob_edge, 0.)

        optimizer_mask.zero_grad()
        loss_mask.backward()
        optimizer_mask.step()

        optimizer_mask_edge.zero_grad()
        loss_mask_edge.backward()
        optimizer_mask_edge.step()

        # optimize model
        mask_prob = mask_module.forward(graph, x, args)
        mask_prob_edge, add_edge_graph = mask_module_edge.forward(graph, x, args, add_edges)
        loss, loss_mask, loss_mask_edge, loss_dict = model(graph, add_edge_graph, x, epoch, args, mask_prob, mask_prob_edge, 0.)

        l2_norm_loss = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + 5e-5 * l2_norm_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")

    return model


def main(args):
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    use_scheduler = args.scheduler

    graph, (num_features, num_classes), add_edges = load_dataset(dataset_name)

    args.num_features = num_features
    args.max_degree = graph.in_degrees().max() + 1

    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]

    torch.cuda.set_device(args.device)
    device = torch.device("cuda:" + str(args.device))

    acc_list = []
    estp_acc_list = []

    for i, seed in enumerate(seeds):
        print(f"# Run {seed} for seeds {seeds} #")
        set_random_seed(seed)

        if dataset_name in ["texas", "chameleon", "squirrel", "actor", "wisconsin", "cornell", "chameleon-filtered", "squirrel-filtered"]:
            num_splits = train_mask.shape[1]
            cur_split = 0 if (num_splits == 1) else (i % num_splits)

            graph.ndata['train_mask'] = train_mask[:, cur_split]
            graph.ndata['val_mask'] = val_mask[:, cur_split]
            graph.ndata['test_mask'] = test_mask[:, cur_split]

        # Model
        model, mask_module, mask_module_edge = build_model(args)
        model.to(device)
        mask_module.to(device)
        mask_module_edge.to(device)
        optimizer = create_optimizer("adam", model, lr, weight_decay)
        optimizer_mask = create_optimizer("adam", mask_module, 0.001, weight_decay)
        optimizer_mask_edge = create_optimizer("adam", mask_module_edge, 0.001, weight_decay)

        if use_scheduler:
            scheduler = lambda epoch: (1 + np.cos(epoch * np.pi / max_epoch)) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
            
        x = graph.ndata["feat"]
        if not load_model:
            model = pretrain(model, mask_module_edge, optimizer_mask_edge, graph, x, optimizer, max_epoch, device, scheduler, args, mask_module, optimizer_mask, add_edges)



        if load_model:
            model = torch.load("outputs/pretrain/" + f'{args.dataset}-{seed}.pt')

        if save_model:
            torch.save(model, '{}{}-{}.pt'.format("outputs/pretrain/", args.dataset, seed))


        model = model.to(device)
        model.eval()


        final_acc, estp_acc = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True)

        print(f"final_acc_at_seed_{seed}: {final_acc: .4f}")
        print(f"final_estp_acc_at_seed_{seed}: {estp_acc: .4f}")


        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)



    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    final_estp_acc, final_estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)

    acc_list = ['{:.4g}'.format(num) for num in acc_list]
    estp_acc_list = ['{:.4g}'.format(num) for num in estp_acc_list]
    print(f"acc_list: {acc_list}")
    print(f"estp_acc_list: {estp_acc_list}")

    print(args)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {final_estp_acc:.4f}±{final_estp_acc_std:.4f}")






if __name__ == "__main__":
    args = build_args()
    args = load_best_configs(args, "configs.yml")

    print(args)
    main(args)

