from model.model import TnpModel

args = {}

# ## path settings
# args["dir"] = str(self.view.dataDir.text())
# args["frames"] = str(self.view.framesDir.text())

# ## detection settings
# args["detection"] = str(self.view.detectionSelector.currentText())
# args["detConf"] = float(self.view.detConfidence.text())
# args["NMS"] = float(self.view.nmsInput.text())
# args["display"] = "False"

## prediction settings
args["predAlgo"] = "Traphic"
args["pretrainEpochs"] = 6
args["trainEpochs"] = 10
args["batch_size"] = 64
args["dropout"] = .5
args["optim"] = "Adam"
args["lr"] = .0001
args["cuda"] = True
args["maneuvers"] = False
args["modelLoc"] = "resources/trained_models/Traphic_model.tar"
args["pretrain_loss"] = ''
args['train_loss'] = "MSE"
args["dir"] = 'resources/data/TRAF'

model = TnpModel()

# model.train(args)
model.evaluate(args)
