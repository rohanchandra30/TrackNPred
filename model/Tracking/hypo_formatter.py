import numpy as np
import os
import pandas as pd

## Indices for hypo frames
HYP_FRAMES_IDX = 0
## Indices for hypo vehicle ids
HYP_VID_IDX = 1
## Indices for hypo top left coords
HYP_TL_X = 2
HYP_TL_Y = 3

## Same as above but for the formatted matrix
FMT_DSET_IDX = 0
FMT_VID_IDX = 1
FMT_FRAMES_IDX = 2
FMT_TL_X = 3
FMT_TL_Y = 4

def getDsetID(fname):
    ## filenames follow 'noisy_hypotheses_xxx.txt'. Interested in the xx but do not
    ## want to make assumptions on xx length
    print(fname)
    return fname.split('_')[2].split('.')[0]

# returns matrix with columns
# Dset id,vehicle id,frame number,tl x,tl y
def formatHypo(dsetID, hypo_mtrx):

    # formatted_mtrx = np.zeros((hypo_mtrx.shape[0], 5))
    formatted_df = {'dset_idx':[], 'vid_idx': [], 'frames_idx': [], 'tl-X': [], 'tl-Y':[]}
    formatted_df['dset_idx'] = list(np.ones(hypo_mtrx.shape[0]).astype(int) * int(dsetID))
    formatted_df['vid_idx'] = list(hypo_mtrx[:,HYP_VID_IDX].astype(int))
    formatted_df['frames_idx'] = list(hypo_mtrx[:,HYP_FRAMES_IDX].astype(int))
    formatted_df['tl-X'] = list(hypo_mtrx[:,HYP_TL_X])
    formatted_df['tl-Y'] = list(hypo_mtrx[:,HYP_TL_Y])


    return pd.DataFrame(formatted_df)

## Creates formatted dataframe 
def getFormattedDF(raw_fname, dsetID):

    with open(raw_fname) as f:
        hypo_mtrx = np.loadtxt(f, delimiter=',')

    return formatHypo(dsetID, hypo_mtrx)


def formatFile(inFilePath, dsetID, outFilePath):
    formatted = getFormattedDF(inFilePath, dsetID)
    formatted.to_csv(outFilePath, index=False, header=False)


def formatFolder(inFolderPath, outFolderPath):
    for fname in os.listdir(inFolderPath):
        formatFile(os.path.join(inFolderPath,fname),
        os.path.join(outFolderPath, fname))
