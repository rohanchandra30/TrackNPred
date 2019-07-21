from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from view.view import TrackNPredView
from control.controller import Controller
from model.model import TnpModel

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()

    controller = Controller()
    view = TrackNPredView()
    ## to get attributes
    view.setupUi(Dialog)
    model = TnpModel(controller)
    controller.setView(view)
    controller.setModel(model)

    #show ui
    Dialog.show()
    sys.exit(app.exec_())