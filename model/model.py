import re
import os
import subprocess
import torch
import argparse
# from model.Detection.Yolo.yolo import detect

from model.Tracking.generate_features import gen_feats
from model.Tracking.DensePeds import densepeds
from model.Tracking.hypo_formatter import formatFile
from model.Tracking.import_data import import_data, merge_n_split
from model.Prediction.traphic import traphicNet
from model.Prediction.social import highwayNet
from model.Prediction.utils import ngsimDataset
from model.Prediction.traphicEngine import TraphicEngine
from model.Prediction.socialEngine import SocialEngine
from torch.utils.data import DataLoader
import datetime
from model.Prediction.sganTrain import main as sganTrain
from model.Prediction.sganEval import main as sganEval
# from model.Prediction.sganArgs import argss as sgan_args

# from model.Detection.Yolo.yolo_gpu import detect
# from sganArgs import args as sgan_args
# from litmethods import sgan_master
# from sganTrain import main as sganMain

GEN_FEATS = "model/Tracking/generate_features.py"
DENS_PEDS = "model/Tracking/DensePeds.py"
HYP_FNAME = "hypotheses.txt"
FORMATTED_HYP_FNAME = "formatted_hypo.txt"
HOMO_FORMAT = "TRAF{}_H.txt"
PRED_INPUT_FORMAT = "{}.npy"
SGAN_INPUT_FORMAT = "{}.txt"
# DATA_LOC = 

sgan_args =argparse.Namespace()

sgan_args.dataset_name='TRAF'
sgan_args.delim = ' '
sgan_args.loader_num_workers = 4
sgan_args.obs_len = 8
sgan_args.pred_len = 12 
sgan_args.skip = 1
sgan_args.batch_size = 64
sgan_args.num_iterations = 10
sgan_args.num_epochs = 20
sgan_args.embedding_dim = 64
sgan_args.num_layers = 1
sgan_args.dropout = 0
sgan_args.batch_norm = 0
sgan_args.mlp_dim = 1024
sgan_args.encoder_h_dim_g = 64
sgan_args.decoder_h_dim_g = 128
sgan_args.noise_dim = None
sgan_args.noise_type =  'gaussian'
sgan_args.noise_mix_type = 'ped'
sgan_args.clipping_threshold_g = 0
sgan_args.g_learning_rate = 5e-4
sgan_args.g_steps =  1
sgan_args.pooling_type =  'pool_net'
sgan_args.pool_every_timestep = 1
sgan_args.bottleneck_dim = 1024
sgan_args.neighborhood_size = 2.0
sgan_args.grid_size = 8
sgan_args.d_type = 'local'
sgan_args.encoder_h_dim_d = 64
sgan_args.d_learning_rate = 5e-4 
sgan_args.d_steps = 2
sgan_args.clipping_threshold_d = 0
sgan_args.l2_loss_weight = 0
sgan_args.best_k = 1
sgan_args.output_dir = os.path.join(os.getcwd(),"resources/trained_models")
sgan_args.print_every = 1
sgan_args.checkpoint_every = 20
sgan_args.checkpoint_name = 'checkpoint'
sgan_args.checkpoint_start_from = None
sgan_args.restore_from_checkpoint = 0
sgan_args.num_samples_check = 5000
sgan_args.use_gpu = 1
sgan_args.timing = 0
sgan_args.gpu_num = "0"
sgan_args.model_path = os.path.join(os.getcwd(),"resources/trained_models/checkpoint_with_model.pt")
sgan_args.num_samples = 20
sgan_args.dset_type = 'test'




class TnpModel:

    def __init__(self, controller = None):
        self.controller = controller

    def MASK_detect(self, inputDir, inputFile, framesDir, outputPath, outputFolder, conf, nms, cuda, thread=None):
        from model.Detection.Mask.mrcnn_detect import detect as mask_detect
        """
        for yolo tracking, takes inputDir (ex. "resources/data/TRAFxx")
        framesDIr (directory containing frames)
        outputTxt: output path of textfile
        """
        mask_detect(inputDir, inputFile, framesDir, outputPath, outputFolder, conf, nms, cuda, thread)

    def YOLO_detect(self, inputDir, inputFile, framesDir, outputPath, outputFolder, conf, nms, cuda, thread=None):
        from model.Detection.Yolo.yolo_gpu import detect as yolo_detect
        """
        for yolo tracking, takes inputDir (ex. "resources/data/TRAFxx")
        framesDIr (directory containing frames)
        outputTxt: output path of textfile
        """
        yolo_detect(inputDir, inputFile, framesDir, outputPath, outputFolder, conf, nms, cuda, thread)

    def tracking(self, dataDir, data_folder, display=True, thread=None):
        if thread:
            thread.signalCanvas("\n[INFO] Generating Features...")
        
        densp_file_path = os.path.join(os.path.join(dataDir, data_folder), "densep.npy")
        if(os.path.exists(densp_file_path)):
            if(thread):
                line = "\n[INFO]: Found {}. Delete this file to generate features again for densepeds.".format(densp_file_path)
                thread.signalCanvas(line)
        else:
            gen_feats(dataDir, data_folder, thread)


        if thread:
            thread.signalCanvas("\n[INFO] Running Densepeds...")

        densepeds(dataDir, data_folder, display, thread)

    def format(self, dataDir, data_folder, thread=None):
        dsetId = re.search(r'\d+', data_folder).group()
        fileName = data_folder
        data_folder = os.path.join(dataDir, data_folder)
        hyp_file = os.path.join(data_folder, HYP_FNAME)
        formatted_hypo = os.path.join(data_folder, FORMATTED_HYP_FNAME)
        homo_file = os.path.join(data_folder, HOMO_FORMAT.format(dsetId))
        pred_input = os.path.join(data_folder, PRED_INPUT_FORMAT.format(fileName))
        sgan_input = os.path.join(data_folder, SGAN_INPUT_FORMAT.format(fileName))

        formatFile(hyp_file, dsetId, formatted_hypo) 
        import_data(formatted_hypo, homo_file, pred_input, sgan_input)

    def getPredArgs(self, viewArgs):
        ## Arguments
        args = {}
        args["batch_size"] = viewArgs["batch_size"]
        args["pretrainEpochs"] = viewArgs["pretrainEpochs"]
        args["trainEpochs"] = viewArgs["trainEpochs"]
        args['cuda'] = viewArgs["cuda"]
        # args['cuda'] = False
        args['modelLoc'] = viewArgs['modelLoc']

        # Network Arguments
        args['dropout_prob'] = viewArgs["dropout"]
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
        args['use_maneuvers'] = viewArgs["maneuvers"]
        args['ours'] = False
        args['nll_only'] = True
        args["learning_rate"] = viewArgs["lr"]
        
        currentDT = datetime.datetime.now()
        args['name'] = "{}_{}_model.tar".format(viewArgs["predAlgo"], currentDT.strftime("%Y_%m_%d_%H_%M"))
        args["pretrain_loss"] = viewArgs['pretrain_loss']
        args['train_loss'] = viewArgs['train_loss']

        return args


    def train(self, viewArgs, thread=None):
        if thread:
            thread.signalCanvas("\n[INFO] Training started...")

        args = self.getPredArgs(viewArgs)
        args['eval'] = False
        predAlgo = viewArgs["predAlgo"]
        optimSelection = viewArgs["optim"]

        if predAlgo == "Traphic":
            if thread:
                thread.signalCanvas("\n[INFO]: Using TRAPHIC model")
            args["ours"] = True
            args['train_flag'] = True
            net = traphicNet(args)
        elif predAlgo == "Social GAN":
            sgan_args.num_epochs = int(args["pretrainEpochs"]) + int(args["trainEpochs"])
            sgan_args.batch_size = args["batch_size"]
            sgan_args.dropout = args['dropout_prob']
            sgan_args.g_learning_rate = args["learning_rate"]
            sgan_args.g_learning_rate = args["learning_rate"]
            if thread:
                thread.signalCanvas("\n[INFO]: Using Sgan model")
                thread.signalCanvas("\n[INFO]: *** Training Prediction Model ***")
            sganTrain(sgan_args, thread)
            return 
        elif predAlgo == "Social-LSTM":
            print(predAlgo)
        elif predAlgo == "Social Conv":
            if thread:
                thread.signalCanvas("\n[INFO]: Using Convolutional Social Pooling")
            args['train_flag'] = True
            net = highwayNet(args)

        if args["cuda"]:
            if thread:
                thread.signalCanvas("\n[INFO]: Using CUDA")
            net.cuda()

        if optimSelection == "Adam":
            optim = torch.optim.Adam(net.parameters(),lr=args['learning_rate'])
            if thread:
                thread.signalCanvas("\n[INFO]: Optimizer: \n" + str(optim))
        else:
            if thread:
                thread.signalCanvas("\n[INFO]: NOT YET IMPLEMENTED")
            return

        crossEnt = torch.nn.BCELoss()
        if thread:
            thread.signalCanvas("\n[INFO]: Loss: \n" + str(crossEnt))

        
        # name of 
        dataset_name = viewArgs["dir"].split('/')[2]
        prediction_data_path = 'model/Prediction/data/{}'.format(dataset_name)
        trSet_path = os.path.join(prediction_data_path, "TrainSet.npy")
        valSet_path = os.path.join(prediction_data_path, "ValSet.npy")
        trSet = ngsimDataset(trSet_path)
        valSet = ngsimDataset(valSet_path)

        trDataloader = DataLoader(trSet,batch_size=args['batch_size'],shuffle=True,num_workers=8,collate_fn=trSet.collate_fn)
        valDataloader = DataLoader(valSet,batch_size=args['batch_size'],shuffle=True,num_workers=8,collate_fn=valSet.collate_fn)

        if predAlgo == "Traphic":
            args["ours"] = True
            engine = TraphicEngine(net, optim, trDataloader, valDataloader, args, thread)
        elif predAlgo == "Social Conv":
            engine = SocialEngine(net, optim, trDataloader, valDataloader, args, thread)
        else:
            if thread:
                thread.signalCanvas("\n[INFO]: NOT YET IMPLEMENTED")
        if thread:
            thread.signalCanvas("\n[INFO]: *** Training Prediction Model ***")
        engine.start()
        

    def evaluate(self, viewArgs, thread=None):
        if thread:
            thread.signalCanvas("\n[INFO] Evaluation started...")
        args = self.getPredArgs(viewArgs)
        args['eval'] = True
        predAlgo = viewArgs["predAlgo"]
        optimSelection = viewArgs["optim"]

        if predAlgo == "Traphic":
            if thread:
                thread.signalCanvas("\n[INFO]: Using Traphic for the saved model")
            args['train_flag'] = False
            args["ours"] = True
            net = traphicNet(args)
        elif predAlgo == "Social GAN":
            sganEval(sgan_args, thread)
            return 
        elif predAlgo == "Social-LSTM":
            print(predAlgo)
        elif predAlgo == "Social Conv":
            if thread:
                thread.signalCanvas("\n[INFO]: Using Convolutional Social Pooling")
            args['train_flag'] = False
            net = highwayNet(args)


        net.eval()
        d = os.path.join(args['modelLoc'])

        if thread:
            thread.signalCanvas(d)

        if os.path.exists(d):
            net.load_state_dict(torch.load(d))
            if thread:
                thread.signalCanvas("\n[INFO]: model loaded")
        else:
            if thread:
                thread.signalCanvas("\n[INFO]: can not find model to evaluate")

        if args["cuda"]:
            if thread:
                thread.signalCanvas("\n[INFO]: Using CUDA")
            net.cuda()

        if optimSelection == "Adam":
            optim = torch.optim.Adam(net.parameters(),lr=args['learning_rate'])
            if thread:
                thread.signalCanvas("\n[INFO]: Optimizer: \n" + str(optim))
        else:
            if thread:
                thread.signalCanvas("\n[INFO]: NOT YET IMPLEMENTED")
            return

        crossEnt = torch.nn.BCELoss()
        if thread:
            thread.signalCanvas("\n[INFO]: Loss: \n" + str(crossEnt))


        # TODO: More hardcodes
        dataset_name = viewArgs["dir"].split('/')[2]
        prediction_data_path = 'model/Prediction/data/{}'.format(dataset_name)
        trSet_path = os.path.join(prediction_data_path, "TrainSet.npy")
        valSet_path = os.path.join(prediction_data_path, "ValSet.npy")
        tstSet_path = os.path.join(prediction_data_path, "TestSet.npy")

        trSet = ngsimDataset(trSet_path)
        trDataloader = DataLoader(trSet,batch_size=args['batch_size'],shuffle=True,num_workers=8,collate_fn=trSet.collate_fn)

        testSet = ngsimDataset(valSet_path)
        testDataloader = DataLoader(testSet,batch_size=args['batch_size'],shuffle=True,num_workers=8,collate_fn=testSet.collate_fn)

        valSet = ngsimDataset(tstSet_path)
        valDataloader = DataLoader(valSet,batch_size=args['batch_size'],shuffle=True,num_workers=8,collate_fn=valSet.collate_fn)

        if predAlgo == "Traphic":
            args["ours"] = True
            engine = TraphicEngine(net, optim, trDataloader, testDataloader, args, thread)
        elif predAlgo == "Social Conv":
            engine = SocialEngine(net, optim, trDataloader, testDataloader, args, thread)
        else:
            if thread:
                thread.signalCanvas("\n[INFO]: NOT YET IMPLEMENTED")
        if thread:
            thread.signalCanvas("\n[INFO]: *** Evaluating Prediction Model ***")
                
        engine.eval(testDataloader)
