import argparse
import numpy as np
import os
import torch

from attrdict import AttrDict

from model.Prediction.sgan.data.loader import data_loader
from model.Prediction.sgan.models import TrajectoryGenerator
from model.Prediction.sgan.losses import displacement_error_2, final_displacement_error
from model.Prediction.sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)



def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate_helper_ade(error, seq_start_end):
    sum_ = torch.zeros(len(error[0][0]), device=torch.device("cuda"))
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error, _ = torch.min(_error, dim=0)
        sum_ = sum_.add(_error)
    return sum_


def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                ade.append(displacement_error_2(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))

            ade_sum = evaluate_helper_ade(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)

        des = sum(ade_outer) / (total_traj)
        ade = sum(des) / (args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return des, ade, fde


def main(args, thread=None):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]
    # dataset_names = np.array(['ours/TRAF11', 'ours/TRAF12', 'ours/TRAF15', 'ours/TRAF17', 'ours/TRAF18', 'ours/TRAF19'])
    dataset_names = np.array([args.dataset_name])
    # dataset_names = np.array(['ours/Beijing1', 'ours/Beijing2'])
    # dataset_names = np.array(['ours/NGSIM1', 'ours/NGSIM2', 'ours/NGSIM3', 'ours/NGSIM4', 'ours/NGSIM5', 'ours/NGSIM6'])
    theirs = False
    if theirs:
        num_models = 1
    else:
        num_models = 5
    des = np.empty((dataset_names.shape[0], num_models, 12))
    ades = np.empty((dataset_names.shape[0], num_models))
    fdes = np.empty((dataset_names.shape[0], num_models))
    idxi = 0
    for dset_name in dataset_names:
        idxj = 0
        for path in paths:
            checkpoint = torch.load(path)
            if "_no_model.pt" not in path:
                generator = get_generator(checkpoint)
                _args = AttrDict(checkpoint['args'])
                dset_path = get_dset_path(dset_name, args.dset_type)
                _, loader = data_loader(_args, dset_path)
                # des[idxi, idxj], ades[idxi, idxj], fdes[idxi, idxj] = evaluate(_args, loader, generator, args.num_samples)
                t1, t2, t3 = evaluate(_args, loader, generator, args.num_samples)
                des[idxi, idxj], ades[idxi, idxj], fdes[idxi, idxj] = t1.cpu(), t2.cpu(), t3.cpu()
                idxj += 1
        print('Calculated on Dataset {}'.format(dset_name))
        idxi += 1

    if thread:
    	# thread.signalCanvas("\nPred Len: {}, DEs:".format(_args.pred_len))
    	# thread.signalCanvas("\n{}".format(np.mean(des, axis=0)))
    	thread.signalCanvas("\nPred Len: {}, ADE: {:.2f}, FDE: {:.2f}".format(_args.pred_len, np.mean(ades), np.mean(fdes)))

    print('Pred Len: {}, DEs:'.format(_args.pred_len))
    print(np.mean(des, axis=0))
    print('Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(_args.pred_len, np.mean(ades), np.mean(fdes)))


# if __name__ == '__main__':
#     args = parser.parse_args()
#     main(args)
