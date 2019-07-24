import re
import os
import subprocess
import torch
# from model.Detection.Yolo.yolo import detect
from model.Detection.Yolo.yolo_gpu import detect as yolo_detect
from model.Detection.Mask.mrcnn_detect import detect as mask_detect

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
from model.Prediction.sganArgs import args as sgan_args

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

class TnpModel:

    def __init__(self, controller):
        self.controller = controller

    def MASK_detect(self, inputDir, inputFile, framesDir, outputPath, outputFolder, conf, nms, cuda, thread=None):
        """
        for yolo tracking, takes inputDir (ex. "resources/data/TRAFxx")
        framesDIr (directory containing frames)
        outputTxt: output path of textfile
        """
        mask_detect(inputDir, inputFile, framesDir, outputPath, outputFolder, conf, nms, cuda, thread)

    def YOLO_detect(self, inputDir, inputFile, framesDir, outputPath, outputFolder, conf, nms, cuda, thread=None):
        """
        for yolo tracking, takes inputDir (ex. "resources/data/TRAFxx")
        framesDIr (directory containing frames)
        outputTxt: output path of textfile
        """
        # detect(inputDir, inputFile, framesDir, outputPath, conf, nms, thread)
        yolo_detect(inputDir, inputFile, framesDir, outputPath, outputFolder, conf, nms, cuda, thread)

    def tracking(self, dataDir, data_folder, display=True, thread=None):
        if thread:
            thread.signalCanvas("\n[INFO] Generating Features...")
        gen_feats(dataDir, data_folder, thread)
        if thread:
            thread.signalCanvas("\n[INFO] Running Densepeds...")

        densepeds(dataDir, data_folder, display, thread)
        # subprocess.call(["python", DENS_PEDS, "--video_folder", data_folder])

    def format(self, dataDir, data_folder, thread=None):
        thread.signalCanvas(data_folder)
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
        predAlgo = viewArgs["predAlgo"]
        optimSelection = viewArgs["optim"]

        if predAlgo == "Traphic":
            if thread:
                thread.signalCanvas("\n[INFO]: Using TRAFPHIC model")
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

        trSet = ngsimDataset('model/Prediction/data/TRAF/TrainSet.npy')
        valSet = ngsimDataset('model/Prediction/data/TRAF/ValSet.npy')

        trDataloader = DataLoader(trSet,batch_size=args['batch_size'],shuffle=True,num_workers=8,collate_fn=trSet.collate_fn)
        valDataloader = DataLoader(valSet,batch_size=args['batch_size'],shuffle=True,num_workers=8,collate_fn=valSet.collate_fn)

        if predAlgo == "Traphic":
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

        predAlgo = viewArgs["predAlgo"]
        optimSelection = viewArgs["optim"]

        if predAlgo == "Traphic":
            if thread:
                thread.signalCanvas("\n[INFO]: Using Traphic for the saved model")
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


        trSet = ngsimDataset('model/Prediction/data/TRAF/TrainSet.npy')
        trDataloader = DataLoader(trSet,batch_size=args['batch_size'],shuffle=True,num_workers=8,collate_fn=trSet.collate_fn)

        testSet = ngsimDataset('model/Prediction/data/TRAF/TestSet.npy')
        testDataloader = DataLoader(testSet,batch_size=args['batch_size'],shuffle=True,num_workers=8,collate_fn=testSet.collate_fn)

        valSet = ngsimDataset('model/Prediction/data/TRAF/ValSet.npy')
        valDataloader = DataLoader(valSet,batch_size=args['batch_size'],shuffle=True,num_workers=8,collate_fn=valSet.collate_fn)

        if predAlgo == "Traphic":
            engine = TraphicEngine(net, optim, trDataloader, testDataloader, args, thread)
        elif predAlgo == "Social Conv":
            engine = SocialEngine(net, optim, trDataloader, testDataloader, args, thread)
        else:
            if thread:
                thread.signalCanvas("\n[INFO]: NOT YET IMPLEMENTED")
        if thread:
            thread.signalCanvas("\n[INFO]: *** Evaluating Prediction Model ***")
                
        engine.eval(testDataloader)