from model.Prediction.trajPredEngine import TrajPredEngine
import torch
import datetime

class TraphicEngine(TrajPredEngine):
    """
    Implementation of abstractEngine for traphic
    TODO:maneuver metrics, too much duplicate code with socialEngine
    """

    def __init__(self, net, optim, train_loader, val_loader, args, thread=None):
        super().__init__(net, optim, train_loader, val_loader, args, thread)
        self.save_name = "traphic"

    def netPred(self, batch):
        hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc, fut, op_mask = batch

        if self.args['cuda']:
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            upp_nbrs = upp_nbrs.cuda()
            mask = mask.cuda()
            upp_mask = upp_mask.cuda()
            lat_enc = lat_enc.cuda()
            lon_enc = lon_enc.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()

        fut_pred  = self.net(hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc)

        return fut_pred

