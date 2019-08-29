import os
import argparse
from model.Prediction.sgan.utils import int_tuple, bool_flag, get_total_norm
from model.Prediction.sgan.utils import relative_to_abs, get_dset_path

argss = {}

parser = argparse.ArgumentParser()

# Dataset options
parser.add_argument('--dataset_name', default='TRAF', type=str)
parser.add_argument('--delim', default=' ')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=12, type=int)
parser.add_argument('--skip', default=1, type=int)

# Optimization
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=10, type=int)
parser.add_argument('--num_epochs', default=20, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)
parser.add_argument('--mlp_dim', default=1024, type=int)

# Generator Options
parser.add_argument('--encoder_h_dim_g', default=64, type=int)
parser.add_argument('--decoder_h_dim_g', default=128, type=int)
parser.add_argument('--noise_dim', default=None, type=int_tuple)
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='ped')
parser.add_argument('--clipping_threshold_g', default=0, type=float)
parser.add_argument('--g_learning_rate', default=5e-4, type=float)
parser.add_argument('--g_steps', default=1, type=int)

# Pooling Options
parser.add_argument('--pooling_type', default='pool_net')
parser.add_argument('--pool_every_timestep', default=1, type=bool_flag)

# Pool Net Option
parser.add_argument('--bottleneck_dim', default=1024, type=int)

# Social Pooling Options
parser.add_argument('--neighborhood_size', default=2.0, type=float)
parser.add_argument('--grid_size', default=8, type=int)

# Discriminator Options
parser.add_argument('--d_type', default='local', type=str)
parser.add_argument('--encoder_h_dim_d', default=64, type=int)
parser.add_argument('--d_learning_rate', default=5e-4, type=float)
parser.add_argument('--d_steps', default=2, type=int)
parser.add_argument('--clipping_threshold_d', default=0, type=float)

# Loss Options
parser.add_argument('--l2_loss_weight', default=0, type=float)
parser.add_argument('--best_k', default=1, type=int)

# Output
parser.add_argument('--output_dir', default=os.path.join(os.getcwd(),"resources/trained_models"))
parser.add_argument('--print_every', default=1, type=int)
parser.add_argument('--checkpoint_every', default=20, type=int)
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=0, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)



# Eval
parser.add_argument('--model_path', default=os.path.join(os.getcwd(),"resources/trained_models/checkpoint_with_model.pt"))
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)



argss = parser.parse_args()

# # Dataset options
# args['dataset_name'] = 'ours/TRAF11'
# args['delim'] = ' '
# args['loader_num_workers'] = 4
# args['obs_len'] = 8
# args['pred_len'] = 12
# args['skip'] = 1

# # Optimization
# args['batch_size'] = 64
# args['num_iterations'] = 10000
# args['num_epochs'] = 200

# # Model Options
# args['embedding_dim'] = 64
# args['num_layers'] = 1
# args['dropout'] = 0
# args['batch_norm'] = 0
# args['mlp_dim'] = 1024

# # Generator Options
# args['encoder_h_dim_g'] = 64
# args['decoder_h_dim_g'] = 128
# args['noise_dim'] = None
# args['noise_type'] = 'gaussian'
# args['noise_mix_type'] = 'ped'
# args['clipping_threshold_g'] = 0
# args['g_learning_rate'] = 5e-4
# args['g_steps'] = 1

# # Pooling Options
# args['pooling_type'] = 'pool_net'
# args['pool_every_timestep'] = 1

# # Pool Net Option
# args['bottleneck_dim'] = 1024

# # Social Pooling Options
# args['neighborhood_size'] = 2.0
# args['grid_size'] = 8

# # Discriminator Options
# args['d_type'] = 'local'
# args['encoder_h_dim_d'] = 64
# args['d_learning_rate'] = 5e-4
# args['d_steps'] = 2
# args['clipping_threshold_d'] = 0

# # Loss Options
# args['l2_loss_weight'] = 0
# args['best_k'] = 1

# # Output
# args['output_dir'] = os.getcwd()
# args['print_every'] = 5
# args['checkpoint_every'] = 100
# args['checkpoint_name'] = 'checkpoint'
# args['checkpoint_start_from'] = None
# args['restore_from_checkpoint'] = 0
# args['num_samples_check'] = 5000

# # Misc
# args['use_gpu'] = 1
# args['timing'] = 0
# args['gpu_num'] = "0"