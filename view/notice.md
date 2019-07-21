# note:
please repalce the line 

'''
class Ui_Dialog(object):
'''

with 

'''
# <<< start here
class TrackNPredView(object):

    def __init__(self):
        self.trainThread = None

    def setTrainThread(self, trainThread):
        self.trainThread = trainThread
# >>>
'''