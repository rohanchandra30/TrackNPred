from model.Detection.Mask.mrcnn_detect import detect 
import os

inputDir = "resources/data/TRAF/TRAF12"
inputFile = "TRAF12.mp4"
framesDir = "frames"
outputPath = os.path.join(inputDir, "det.txt")
outputFolder = os.path.join(inputDir, "detFrames")

detect(inputDir, inputFile, framesDir, outputPath, outputFolder, .8, .8, True)
