from argparse import Namespace
import logging
from typing import Callable, List, Union

import numpy as np

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import trange

from chemprop.data import MoleculeDataset
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR, SinexpLR


def train(model: nn.Module,
          data: Union[MoleculeDataset, List[MoleculeDataset]],
          loss_func: Callable,
          metric_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: Namespace,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data: A MoleculeDataset (or a list of MoleculeDatasets if using moe).
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print
    
    model.train()
    
    data.shuffle()

    loss_sum, metric_sum, iter_count = [0]*(len(args.atom_targets) + len(args.bond_targets)), \
                                       [0]*(len(args.atom_targets) + len(args.bond_targets)), 0

    num_iters = len(data) // args.batch_size * args.batch_size  # don't use the last batch if it's small, for stability

    iter_size = args.batch_size

    for i in trange(0, num_iters, iter_size):
        # Prepare batch
        if i + args.batch_size > len(data):
            break
        mol_batch = MoleculeDataset(data[i:i + args.batch_size])
        smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()
        batch = smiles_batch
        #mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])

        # FIXME assign 0 to None in target
        # targets = [[0 if x is None else x for x in tb] for tb in target_batch]


        targets = [torch.Tensor(np.concatenate(x)) for x in zip(*target_batch)]
        if next(model.parameters()).is_cuda:
        #   mask, targets = mask.cuda(), targets.cuda()
            targets = [x.cuda() for x in targets]
        # FIXME
        #class_weights = torch.ones(targets.shape)

        #if args.cuda:
        #   class_weights = class_weights.cuda()

        # Run model
        model.zero_grad()
        preds = model(batch, features_batch)
        targets = [x.reshape([-1, 1]) for x in targets]

        #FIXME mutlticlass
        '''
        if args.dataset_type == 'multiclass':
            targets = targets.long()
            loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * class_weights * mask
        else:
            loss = loss_func(preds, targets) * class_weights * mask
        '''
        loss_multi_task = []
        metric_multi_task = []
        for target, pred in zip(targets, preds):
            loss = loss_func(pred, target)
            loss = loss.sum() / target.shape[0]
            loss_multi_task.append(loss)
            if args.cuda:
                metric = metric_func(pred.data.cpu().numpy(), target.data.cpu().numpy())
            else:
                metric = metric_func(pred.data.numpy(), target.data.numpy())
            metric_multi_task.append(metric)

        loss_sum = [x + y for x,y in zip(loss_sum, loss_multi_task)]
        iter_count += 1

        sum(loss_multi_task).backward()
        optimizer.step()

        metric_sum = [x + y for x,y in zip(metric_sum, metric_multi_task)]

        if isinstance(scheduler, NoamLR) or isinstance(scheduler, SinexpLR):
            scheduler.step()

        n_iter += len(mol_batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = [x / iter_count for x in loss_sum]
            metric_avg = [x / iter_count for x in metric_sum]
            loss_sum, iter_count, metric_sum = [0]*(len(args.atom_targets) + len(args.bond_targets)), \
                                               0, \
                                               [0]*(len(args.atom_targets) + len(args.bond_targets))

            loss_str = ', '.join(f'lss_{i} = {lss:.4e}' for i, lss in enumerate(loss_avg))
            metric_str = ', '.join(f'mc_{i} = {mc:.4e}' for i, mc in enumerate(metric_avg))
            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            debug(f'{loss_str}, {metric_str}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')

            if writer is not None:
                for i, lss in enumerate(loss_avg):
                    writer.add_scalar(f'train_loss_{i}', lss, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)

    return n_iter
