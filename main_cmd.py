from model.model import TnpModel
from model.utils import *
from model.Tracking.hypo_formatter import formatFile
from model.Tracking.import_data import import_data, merge_n_split

import argparse
import warnings
warnings.filterwarnings("ignore")

VERBOSE = True


# default values for running the trackNPred, you can also specify it in cmd parser

# the dataset you want to run
# DSET_IDS = [12]


# dataset dirs
DATA_DIR = 'resources/data/TRAF'
PRED_DATA_DIR = 'model/Prediction/data/TRAF'


# enable/disable each part
DETECTION = True
TRACKING = True
FORMATTING = True
TRAIN = True
EVAL = True

# training option
FRAMES = 'frames'
DETALGO = 'YOLO'
DETCONF = 0.5
NMS = 0.4
PREDALGO = 'Traphic'
PRETRAINEPOCHS= 6
TRAINEPOCHS= 10
BATCH_SIZE = 128
DROPOUT = 0.5
OPTIM= 'Adam'
LEARNING_RATE= 0.001
CUDA= True
MANEUVERS = False
MODELLOC= "model/Prediction/trained_models"
PRETRAIN_LOSS = 'MSE'
TRAIN_LOSS = 'NLL'


# do not change this unless you want to run other dataset other than TRAF
DATA_FOLDER = 'TRAF{}'
VIDEO = 'TRAF{}.mp4'
PRED_FILE = 'TRAF{}.npy'
HOMO = 'TRAF{}_H.txt'



if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="TrackNPred command line control")

	parser.add_argument('--dir', help="location of the dataset for tracking", default=DATA_DIR)
	parser.add_argument('--predir', help="location of the dataset for trajectory prediction, result of tracking",
		default=PRED_DATA_DIR)
	parser.add_argument('--detection', '-d', help='enable detection step', default=DETECTION, type=bool)
	parser.add_argument('--tracking', '-track', help='enable tracking step', default=TRACKING, type=bool)
	parser.add_argument('--formatting', '-f', help='enable formatting step', default=FORMATTING, type=bool)
	parser.add_argument('--train', '-t', help='enable train step', default=TRAIN, type=bool)
	parser.add_argument('--eval', help='enable evaluation step', default=EVAL, type=bool)
	parser.add_argument('--frames', help="location of the frames of the data in each dataset folder", default=FRAMES)
	parser.add_argument('--detalgo', help="detection method", default=DETALGO)
	parser.add_argument('--conf', help='confidence in tracking', default=DETCONF)
	parser.add_argument('--nms', help='nms in tracking',default=NMS)
	parser.add_argument('--predalgo', help='prediction algorithm', default=PREDALGO)
	parser.add_argument('--pretrainEpochs', help='number of epochs for pretraining', default=PRETRAINEPOCHS)
	parser.add_argument('--trainEpochs', '-e', help='number of epochs for training', default=TRAINEPOCHS)
	parser.add_argument('--batch_size', '-b', help='bastch size', default=BATCH_SIZE)
	parser.add_argument('--dropout', help='dropout probability', default=DROPOUT)
	parser.add_argument('--optim', help='optimiser', default=OPTIM)
	parser.add_argument('--lr', help='learning rate', default=LEARNING_RATE)
	parser.add_argument('--cuda', '-g', help='GPU option', default=CUDA, type=bool)
	parser.add_argument('--maneuvers', help='maneuvers option', default=MANEUVERS, type=bool)
	parser.add_argument('--modelLoc', help='trained prediction store/load location', default=MODELLOC)
	parser.add_argument('--pretrain_loss', help='pretrain loss algorithm', default=PRETRAIN_LOSS)
	parser.add_argument('--train_loss', help='train loss algorithm', default=TRAIN_LOSS)
	parser.add_argument('--list', '-l', action='append', help='DATASet', required=True)

	args = parser.parse_args()

	model = TnpModel(None)

	file_names = []

	for i in args.list:


		folder = os.path.join(args.dir, DATA_FOLDER.format(i))
		video = VIDEO.format(i)
		det = 'det.txt'
		if args.detection: 
			sayVerbose(VERBOSE, "begin detection for {}...".format(folder))
			model.YOLO_detect(folder, video, args.frames, det, "detectedFrames", args.conf, args.nms, args.cuda)    
			sayVerbose(VERBOSE, "finished detection for {}...".format(folder))

		if args.tracking:
			sayVerbose(VERBOSE, "begin tracking for {}...".format(folder))
			model.tracking(args.dir, DATA_FOLDER.format(i), False)
			sayVerbose(VERBOSE, "finished tracking for {}...".format(folder))
	    
		hypo = os.path.join(folder, 'hypotheses.txt')
		formatted_hypo = os.path.join(folder, 'formatted_hypo.txt')
		homo = os.path.join(folder, HOMO.format(i))
		out = os.path.join(folder, PRED_FILE.format(i))

		file_names.append(out)
		
		if args.tracking:
			sayVerbose(VERBOSE, "Formatting {} for prediction...".format(folder))
			formatFile(hypo, i, formatted_hypo) 
			import_data(formatted_hypo, homo, out)
			sayVerbose(VERBOSE, "Done formatting for {}... ".format(folder))

	pred_data = os.path.join(args.predir, PRED_FILE)

	merge_n_split(file_names, pred_data)
	sayVerbose(VERBOSE, "Done merging data for training.")


	if args.train:
		sayVerbose(VERBOSE, "Start training...")
		model.train(args)
		sayVerbose(VERBOSE, "Done training.")


	if args.eval:
		sayVerbose(VERBOSE, "Start evaluating...")
		model.evaluate(args)
		sayVerbose(VERBOSE, "Done evaluating.")

