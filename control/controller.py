import os
from control.trainThread import TrainThread
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QSize


class Controller:

    def __init__(self):
        self.view = None

    def connectControllers(self):
        ## connect train button
        self.view.trainButton.clicked.connect(self.handleTrain)
        # connect eval button
        self.view.evalButton.clicked.connect(self.handleEval)
        ## clear
        self.view.clearButton.clicked.connect(self.clear)
        ## stop
        self.view.stopButton.clicked.connect(self.stop)

        self.view.trackingFlag.stateChanged.connect(self.toggleTrackBox)
        self.view.predictionFlag.stateChanged.connect(self.togglePredictionBox)


    def stop(self):
        self.addToCanvas("Stopping...")
        self.view.stopButton.setEnabled(False)
        self.view.trainThread.terminate() 
        self.enableEverything()
        self.updateBotLabel("")
        self.updateTopLabel("")
        self.setBotBar(0)
        self.setTopBar(0)


    def setView(self, view):
        self.view = view
        self.connectControllers()

    def setModel(self, model):
        self.model = model

    def getArgs(self):
        """"
        TODO: error handling
        """
        args = {}

        ## path settings
        args["dir"] = str(self.view.dataDir.text())
        args["frames"] = str(self.view.framesDir.text())

        ## detection settings
        args["detection"] = str(self.view.detectionSelector.currentText())
        args["detConf"] = float(self.view.detConfidence.text())
        args["NMS"] = float(self.view.nmsInput.text())
        
        ## prediction settings
        args["predAlgo"] = str(self.view.predictionSelect.currentText())
        args["pretrainEpochs"] = int(self.view.pretrainEpochs.text())
        args["trainEpochs"] = int(self.view.trainEpochs.text())
        args["batch_size"] = int(self.view.batchSize.text())
        args["dropout"] = float(self.view.dropout.text())
        args["optim"] = str(self.view.optimizer.currentText())
        args["lr"] = float(self.view.learningRate.text())
        args["cuda"] = bool(self.view.cuda.isChecked())
        args["maneuvers"] = bool(self.view.maneuvers.isChecked())
        args["modelLoc"] = str(self.view.modelLoc.currentText())
        args["pretrain_loss"] = str(self.view.pretrainLoss.currentText())
        args['train_loss'] = str(self.view.trainLoss.currentText())
        args["display"] = bool(self.view.display.isChecked())

        
        return args

    def handleTrain(self):
        # process gui
        self.view.trainButton.setEnabled(False)
        self.view.evalButton.setEnabled(False)
        self.view.tracking_box.setEnabled(False)
        self.view.prediction_box1.setEnabled(False)
        self.view.prediction_box2.setEnabled(False)
        self.view.dataDir.setEnabled(False)
        self.view.framesDir.setEnabled(False)
        self.view.trackingFlag.setEnabled(False)
        self.view.predictionFlag.setEnabled(False)
        self.view.cuda.setEnabled(False)
        self.updateBotLabel("")
        self.updateTopLabel("")
        self.setBotBar(0)
        self.setTopBar(0)

        args = self.getArgs()
        args["trackingFlag"] = bool(self.view.trackingFlag.isChecked())
        args["predictionFlag"] = bool(self.view.predictionFlag.isChecked())
        args["evaluationFlag"] = False
        self.view.setTrainThread(TrainThread(args, self.model, self))
        self.view.stopButton.setEnabled(True)

        # connect signals
        self.view.trainThread.pbar2LabelSig.connect(self.updateBotLabel)
        self.view.trainThread.pbar1LabelSig.connect(self.updateTopLabel)
        self.view.trainThread.pbar2Sig.connect(self.setBotBar)
        self.view.trainThread.pbar1Sig.connect(self.setTopBar)
        self.view.trainThread.canvasSig.connect(self.addToCanvas)
        self.view.trainThread.errorSig.connect(self.handleError)
        self.view.trainThread.imgSig.connect(self.dispImage)

        self.view.trainThread.start()


    def disableEverything(self):
        # process gui
        self.view.trainButton.setEnabled(False)
        self.view.evalButton.setEnabled(False)
        self.view.tracking_box.setEnabled(False)
        self.view.prediction_box1.setEnabled(False)
        self.view.prediction_box2.setEnabled(False)
        self.view.dataDir.setEnabled(False)
        self.view.framesDir.setEnabled(False)
        self.view.trackingFlag.setEnabled(False)
        self.view.predictionFlag.setEnabled(False)
        self.view.cuda.setEnabled(False)

    def enableEverything(self):
        # process gui
        self.view.trainButton.setEnabled(True)
        self.view.evalButton.setEnabled(True)
        self.view.tracking_box.setEnabled(True)
        self.view.prediction_box1.setEnabled(True)
        self.view.prediction_box2.setEnabled(True)
        self.view.dataDir.setEnabled(True)
        self.view.framesDir.setEnabled(True)
        self.view.trackingFlag.setEnabled(True)
        self.view.predictionFlag.setEnabled(True)
        self.view.cuda.setEnabled(True)

    def handleEval(self):
        self.disableEverything()
        self.updateBotLabel("")
        self.updateTopLabel("")
        self.setBotBar(0)
        self.setTopBar(0)

        args = self.getArgs()
        args["trackingFlag"] = False
        args["predictionFlag"] = False
        args["evaluationFlag"] = True
        self.view.setTrainThread(TrainThread(args, self.model, self))

        # connect signals
        self.view.trainThread.pbar2LabelSig.connect(self.updateBotLabel)
        self.view.trainThread.pbar1LabelSig.connect(self.updateTopLabel)
        self.view.trainThread.pbar2Sig.connect(self.setBotBar)
        self.view.trainThread.pbar1Sig.connect(self.setTopBar)
        self.view.trainThread.canvasSig.connect(self.addToCanvas)
        self.view.trainThread.errorSig.connect(self.handleError)

        self.view.trainThread.start()


    def clear(self):
        self.view.canvas.setText('')
        self.view.topBar.setValue(0)
        self.view.botBar.setValue(0)
        self.view.topBarLabel.setText('')
        self.view.botBarLabel.setText('')


    def toggleTrackBox(self, state):
        if state > 0:
            self.view.tracking_box.setEnabled(True)
        else:
            self.view.tracking_box.setEnabled(False)

    def togglePredictionBox(self, state):
        if state > 0:
            self.view.prediction_box1.setEnabled(True)
            self.view.prediction_box2.setEnabled(True)
        else:
            self.view.prediction_box1.setEnabled(False)
            self.view.prediction_box2.setEnabled(False)

    def updateTopLabel(self, label):
        self.view.topBarLabel.setText(label)

    def updateBotLabel(self, label):
        self.view.botBarLabel.setText(label)


    def dispImage(self, path):
        # print('hello')
        pixmap = QPixmap(path)
        h = self.view.imgDisplay.height()
        w = self.view.imgDisplay.width()
        self.view.imgDisplay.setPixmap(pixmap.scaled(QSize(h, w), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # self.view.imgDisplay.set

    # def addToCanvas(self, text):
    #     curr = self.view.canvas.toPlainText()
    #     new = curr + "{}".format(text)
    #     self.view.canvas.setText(new)
    #     self.view.canvas.verticalScrollBar().setValue(self.view.canvas.verticalScrollBar().maximum())

    def addToCanvas(self, text):
        self.view.canvas.insertHtml("<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; \
            margin-right:0px; -qt-block-indent:0; text-indent:0px;\">{}<br></p>".format(text))
        self.view.canvas.verticalScrollBar().setValue(self.view.canvas.verticalScrollBar().maximum())

    def handleError(self, text):
        # print(self.view.canvas.toHtml())
        self.view.canvas.insertHtml("<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; \
            margin-right:0px; -qt-block-indent:0; text-indent:0px; color:#ff0000;\">{}<br></p>".format(text))
        self.view.canvas.verticalScrollBar().setValue(self.view.canvas.verticalScrollBar().maximum())


    def incrementTop(self, x):
        topVal = self.view.topBar.value()
        self.view.topBar.setValue(topVal + x)

    def setTopBar(self, x):
        self.view.topBar.setValue(x)

    def setBotBar(self, x):
        self.view.botBar.setValue(x)





