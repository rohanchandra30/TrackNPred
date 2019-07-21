import os

""""
TODO: move this to model
"""


def sayVerbose(vb, stmt):
    if vb:
        print(stmt)


def ensure_dir(d):
    if not os.path.exists(d):
        print("making directory: {}".format(d))
        os.makedirs(d)
