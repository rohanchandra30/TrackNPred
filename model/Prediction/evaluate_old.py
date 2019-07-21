from __future__ import print_function
import torch
from traphic import highwayNet
from utils import ngsimDataset,maskedNLL,maskedMSETest,maskedNLLTest
from torch.utils.data import DataLoader
import time
import warnings
warnings.filterwarnings("ignore")

## Network Arguments
args = {}
args['dropout_prob'] = 0.5
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 50
args['grid_size'] = (13,3)
args['upp_grid_size'] = (7,3)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['num_lat_classes'] = 3
args['num_lon_classes'] = 2
args['train_flag'] = False
args['use_maneuvers'] = False
args['model_path'] = 'trained_models/m_false/cslstm_b_pretrain2_NGSIM.tar'
args['dataset_path'] = 'data/TestSetTRAF.mat'
args['ours'] = True

# Evaluation metric:
metric = 'rmse'  #nll or rmse


# Initialize network
net = highwayNet(args)
batch_size = 128
net.load_state_dict(torch.load(args['model_path']),strict=False)
if args['use_cuda']:
    net = net.cuda()

tsSet = ngsimDataset(args['dataset_path'])
tsDataloader = DataLoader(tsSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=tsSet.collate_fn)

lossVals = torch.zeros(args['out_length']).cuda()
counts = torch.zeros(args['out_length']).cuda()


for i, data in enumerate(tsDataloader):
    st_time = time.time()
    hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc, fut, op_mask = data

    # Initialize Variables
    if args['use_cuda']:
        hist = hist.cuda()
        nbrs = nbrs.cuda()
        upp_nbrs = upp_nbrs.cuda()
        mask = mask.cuda()
        upp_mask = upp_mask.cuda()
        lat_enc = lat_enc.cuda()
        lon_enc = lon_enc.cuda()
        fut = fut.cuda()
        op_mask = op_mask.cuda()

    if metric == 'nll':
        # Forward pass
        if args['use_maneuvers']:
            fut_pred, lat_pred, lon_pred = net(hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc)
            l,c = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask)
        else:
            fut_pred = net(hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc)
            l, c = maskedNLLTest(fut_pred, 0, 0, fut, op_mask,use_maneuvers=False)
    else:
        # Forward pass
        if args['use_maneuvers']:
            fut_pred, lat_pred, lon_pred = net(hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc)
            fut_pred_max = torch.zeros_like(fut_pred[0])
            for k in range(lat_pred.shape[0]):
                lat_man = torch.argmax(lat_pred[k, :]).detach()
                lon_man = torch.argmax(lon_pred[k, :]).detach()
                indx = lon_man*3 + lat_man
                fut_pred_max[:,k,:] = fut_pred[indx][:,k,:]
            l, c = maskedMSETest(fut_pred_max, fut, op_mask)
        else:
            fut_pred = net(hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc)
            l, c = maskedMSETest(fut_pred, fut, op_mask)


    lossVals +=l.detach()
    counts += c.detach()

if metric == 'nll':
    print(lossVals / counts)
else:
    rmse = torch.pow(lossVals / counts, 0.5)*(0.3048) # Calculate RMSE and convert from feet to meters
    # print([sum(rmse.reshape(-1,10)[i])/10 for i in range(5)])
    print(rmse)
