import torch
from traphic import traphicNet
from social import highwayNet
from utils import ngsimDataset
from torch.utils.data import DataLoader
import warnings
import math
import numpy as np


from traphicEngine import TraphicEngine
from socialEngine import SocialEngine
from sganArgs import args as sgan_args
# from litmethods import sgan_master
from sganTrain import main as sganMain

# ignite
warnings.filterwarnings("ignore")


verbose = True
# dset = [11, 12]


def runPytorchModel(args):
    # Initialize network
    if args['model'] == "traphic":
        net = traphicNet(args)
    else:
        net = highwayNet(args)

    # net.load_state_dict(torch.load('trained_models/m_false/cslstm_b_pretrain2_NGSIM.tar'), strict=False)

    if args['cuda']:
        print("Using cuda")
        net = net.cuda()
        # net = net.to("cuda")

    ## Initialize optimizer
    optim = torch.optim.Adam(net.parameters(),lr=args['learning_rate'])
    crossEnt = torch.nn.BCELoss()

    if verbose:
        print("*" * 3, "Using model: ", net)
        print("*" * 3, "Optim: ", optim)
        print("*" * 3, "Creating dataset and dataloaders...")

    ## Initialize data loaders
    trSet = ngsimDataset('model/Prediction/data/TRAF/TRAFTrainSet.npy')
    valSet = ngsimDataset('model/Prediction/data/TRAF/TRAFValSet.npy')
    # trSet = ngsimDataset('./Prediction/data/TrainSetTRAF.mat')
    # valSet = ngsimDataset('./Prediction/data/ValSetTRAF.mat')
    trDataloader = DataLoader(trSet,batch_size=args['batch_size'],shuffle=True,num_workers=8,collate_fn=trSet.collate_fn)
    valDataloader = DataLoader(valSet,batch_size=args['batch_size'],shuffle=True,num_workers=8,collate_fn=valSet.collate_fn)

    if verbose:
        print("starting training...")

    if args['model'] == "traphic":
        if verbose:
            print("Training TRAPHIC")
        traphic = TraphicEngine(net, optim, trDataloader, valDataloader, args)
        traphic.start()
    else:
        if verbose:
            print("Training conv social pooling")
        social = SocialEngine(net, optim, trDataloader, valDataloader, args)
        social.start()

def evalPytorchModel(args):
    # Initialize network
    args['eval'] = True
    if args['model'] == "traphic":
        net = traphicNet(args)
    else:
        net = highwayNet(args)

    # net.load_state_dict(torch.load('trained_models/m_false/cslstm_b_pretrain2_NGSIM.tar'), strict=False)

    if args['use_cuda']:
        print("Using cuda")
        net = net.cuda()
        # net = net.to("cuda")

    ## Initialize optimizer
    optim = torch.optim.Adam(net.parameters(),lr=args['learning_rate'])
    crossEnt = torch.nn.BCELoss()

    if verbose:
        print("*" * 3, "Using model: ", net)
        print("*" * 3, "Optim: ", optim)
        print("*" * 3, "Creating dataset and dataloaders...")

    ## Initialize data loaders
    testSet = ngsimDataset('model/Prediction/data/TRAF/TRAFTestSet.npy')
    testDataloader = DataLoader(testSet,batch_size=args['batch_size'],shuffle=True,num_workers=8,collate_fn=testSet.collate_fn)

    if verbose:
        print("evaluating...")


    net.load_state_dict(torch.load("model/Prediction/trained_models/{}_{}.tar".format(args['model'], args['name'])))


    if args['model'] == "traphic":
        if verbose:
            print("Evaluating TRAPHIC")
        traphic = TraphicEngine(net, optim, None, testDataloader, args)
        traphic.start()
    else:
        if verbose:
            print("Evaluating SOCIAL")
        social = SocialEngine(net, optim, None, testDataloader, args)
        social.start()      

def runTFModel(args):
    if args['model'] == "sgan":
        print("Training sgan")
        sganMain(sgan_args)
    elif args['model'] == "social-lstm":
        pass

# # training traphic or social conv
#     if not args.eval_only:
#         if args.model == "traphic" or args.model == "sconv":
#             runPytorchModel(paras)
#         elif args.model == 'sgan':
#             runTFModel(paras)
#         else:
#             print('invalid model, please try again')

#     # eveluation
#     if args.model == "traphic" or args.model == "sconv":
#         evalPytorchModel(paras)
#     elif args.model == 'sgan':
#         runTFModel(paras)
#     else:
#         print('invalid model, please try again')




