from PyQt5.QtCore import QThread, pyqtSignal, Qt
import re
import os
from model.Tracking.import_data import merge_n_split
import traceback


class TrainThread(QThread):

    pbar2LabelSig = pyqtSignal(str)
    pbar1LabelSig = pyqtSignal(str) 
    pbar2Sig = pyqtSignal(int)
    pbar1Sig = pyqtSignal(int)
    canvasSig = pyqtSignal(str)
    errorSig = pyqtSignal(str)
    imgSig = pyqtSignal(str)


    def __init__(self, args, model, controller):
        QThread.__init__(self)
        self.args = args
        self.model = model
        self.controller = controller

    def signalBotLabel(self, txt):
        self.pbar2LabelSig.emit(txt)

    def signalBotBar(self, val):
        self.pbar2Sig.emit(val)

    def signalTopBar(self, val):
        self.pbar1Sig.emit(val)

    def signalTopLabel(self, txt):
        self.pbar1LabelSig.emit(txt)

    def signalCanvas(self, txt):
        # pass
        self.canvasSig.emit(txt)

    def signalError(self, txt):
        self.errorSig.emit(txt)

    def signalImg(self, path):
        self.imgSig.emit(path)

    def detect(self, args):

        self.signalCanvas("\n[INFO] Detection started...")

        dataDir = args["dir"]
        conf = args["detConf"]
        nms = args["NMS"]

        nfolders = len(os.listdir(dataDir))

        if args["detection"] == "YOLO":
            self.signalCanvas("\n[INFO] Using YOLO for detection")
        elif args["detection"] == "MASK":
            self.signalCanvas("\n[INFO] Using MASK for detection")

        for i, data_folder in enumerate(os.listdir(dataDir)):
            input_dir = os.path.join(dataDir, data_folder)
            outputPath = os.path.join(input_dir, "det.txt")

            self.signalBotLabel("Detection: {}/{} folders".format(max(i, 1), nfolders))
            self.signalBotBar(int(i + 1 / nfolders * 100))

            if(os.path.exists(outputPath)):
                self.signalCanvas("\n[INFO]: Detection file (det.txt) found at {}. Delete this file to perform detection for {}".format(outputPath, data_folder))

            else:
                input_file = "{}.mp4".format(data_folder)
                frames = args["frames"]
                outputFolder = os.path.join(input_dir, "detectedFrames")
                os.makedirs(outputFolder, exist_ok=True)

                #    self.updateTopBar(5)
                if args["detection"] == "YOLO":
                    self.model.YOLO_detect(input_dir, input_file, frames, "det.txt", "detectedFrames", conf, nms, args["cuda"], self)
                elif args["detection"] == "MASK":
                    self.model.MASK_detect(input_dir, input_file, frames, "det.txt", "detectedFrames", conf, nms, args["cuda"], self)

        self.signalCanvas("\n[INFO] Detection finished")

    def track(self, args):
        self.signalCanvas("\n[INFO] Tracking started...")
        dataDir = args["dir"]
        nfolders = len(os.listdir(dataDir))

        for i, data_folder in enumerate(os.listdir(dataDir)):
            self.signalBotLabel("Tracking: {}/{} folders".format(max(i, 1), nfolders))
            self.signalBotBar(int(i + 1 / nfolders * 100))
            self.model.tracking(dataDir, data_folder, thread=self)

        self.signalCanvas("\n[INFO] Tracking finished")

    def format(self, args):
        self.signalCanvas("\n[INFO] Formatting hypothesis files...")
        dataDir = args["dir"]
        print("DATADIR: ", dataDir)
        # nfolders = len(os.listdir(dataDir))
        file_names = []
        for i, data_folder in enumerate(os.listdir(dataDir)):
            self.model.format(dataDir, data_folder, self)
            loc = os.path.join(dataDir, data_folder)
            file_names.append(os.path.join(loc, data_folder + '.npy'))

        train_data_dir = "model/Prediction/data/{}".format(dataDir.split('/')[2])
        os.makedirs(train_data_dir, exist_ok=True)
        print("train data dir: ", train_data_dir)
        train_data_dir = train_data_dir + "/{}"
        merge_n_split(file_names, train_data_dir)

        self.signalCanvas("\n[INFO] Done Formatting hypothesis files...")


    def run(self):
        try:
            if self.args['trackingFlag']:
                self.signalBotBar(0)
                self.signalTopBar(0)
                self.detect(self.args)
                self.signalBotBar(0)
                self.signalTopBar(0)
                self.track(self.args)
            if self.args['predictionFlag']:
                self.format(self.args)
                self.signalBotBar(0)
                self.signalTopBar(0)
                self.model.train(self.args, self)
                self.signalCanvas("\n[INFO] Done Training!\n")
            if self.args['evaluationFlag']:
                self.signalBotBar(0)
                self.signalTopBar(0)
                self.model.evaluate(self.args, self)
                self.signalCanvas("\n[INFO] Done Evaluating!\n")
        except Exception as e:
            self.signalError('[ERROR] {}'.format(e))
            for l in traceback.format_exc().split('\n'):
                if len(l) != 0:
                    self.signalError('[ERROR] {}'.format(l))
            # self.signalError('[ERROR] {}'.format(e))
            # for l in traceback.format_exc().split('\n'):
            #     if len(l) != 0:
            #         self.signalError('[ERROR] {}'.format(l))

        finally:
            self.controller.enableEverything()

            self.exec()
