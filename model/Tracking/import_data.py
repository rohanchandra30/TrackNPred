import re
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os

def import_data(file_dir, homography_dir, out_dir, sgan_dir, toFeet=False):

    tranform(file_dir, homography_dir, out_dir, sgan_dir, toFeet)
# Sicne the model uses 3 secs of trajectory history for prediction,
# the initial 3 seconds of each trajectory is not used for training/testing


def merge_n_split(file_names, out_format):

    traj_train = np.array([])
    traj_val = np.array([])
    traj_test = np.array([])

    track_train = defaultdict(dict)
    track_val = defaultdict(dict)
    track_test = defaultdict(dict)

    print("Start spliting data...")

    for f in file_names:
        # print("Reading dataset {}...".format(d))
        d = (int)(re.search(r'\d+', f).group())
        npy_path = f
        # print(npy_path)
        data = np.load(npy_path, allow_pickle=True)

        # constructing train, val and testset for trajectory
        traj = data[0]
        traj_id = np.unique(traj[:,1])

        # split the dataset using vehicle id
        traj_id, test_id = train_test_split(traj_id, test_size=0.2, random_state=2)
        train_id, val_id = train_test_split(traj_id, test_size=0.125, random_state=2)

        # get trajectory with vehicle id
        train = np.array(traj[ [(s in train_id) for s in traj[:,1]] ])
        if traj_train.size == 0:
            traj_train = train
        else:
            traj_train = np.concatenate((traj_train, train), axis=0)

        val = np.array(traj[ [(s in val_id) for s in traj[:,1]] ])
        if traj_val.size == 0:
            traj_val = val
        else:
            traj_val = np.concatenate((traj_val, val), axis=0)

        test = np.array(traj[ [(s in test_id) for s in traj[:,1]] ])
        if traj_test.size == 0:
            traj_test = test
        else:
            traj_test = np.concatenate((traj_test, test), axis=0)


        # constructing train, val and testset for tracks
        track = data[1]

        for i in train_id:
            track_train[d][i] = track[i]

        for i in val_id:
            track_val[d][i] = track[i]

        for i in test_id:
            track_test[d][i] = track[i]     

        print("Dataset {} finsihed.".format(d))



    if not os.path.exists(out_format.format("train")):
        os.makedirs(out_format.format("train"))

    f = open(out_format.format("train/TrainSet.txt"), 'w')
    for line in traj_train:
        # f.write("{}\t{}\t{}\t{}\n".format(int(line[2]), int(line[1]), line[3], line[4]))
        f.write("{}\t{}\t{}\t{}\t{}\n".format(int(line[2]), int(line[1]), line[3], line[4], int(line[0])))
    f.close()

    if not os.path.exists(out_format.format("val")):
        os.makedirs(out_format.format("val"))

    f = open(out_format.format("val/ValSet.txt"), 'w')
    for line in traj_val:
        # f.write("{}\t{}\t{}\t{}\n".format(int(line[2]), int(line[1]), line[3], line[4]))
        f.write("{}\t{}\t{}\t{}\t{}\n".format(int(line[2]), int(line[1]), line[3], line[4], int(line[0])))
    f.close()

    if not os.path.exists(out_format.format("test")):
        os.makedirs(out_format.format("test"))    

    f = open(out_format.format("test/TestSet.txt"), 'w')
    for line in traj_test:
        # f.write("{}\t{}\t{}\t{}\n".format(int(line[2]), int(line[1]), line[3], line[4]))
        f.write("{}\t{}\t{}\t{}\t{}\n".format(int(line[2]), int(line[1]), line[3], line[4], int(line[0])))
    f.close()

    np.save(out_format.format("TrainSet.npy"), np.array([traj_train, track_train]))
    np.save(out_format.format("ValSet.npy"), np.array([traj_val, track_val])) 
    np.save(out_format.format("TestSet.npy"), np.array([traj_test, track_test]))
    print("Training file saved and ready.")

    # print(traj_train)
    # print(traj_val)
    # print(traj_test)
    # print(len(traj_train))
    # print(len(traj_val))
    # print(len(traj_test))
    # for d in 
    # print(len(traj_train))
    # print(len(traj_val))
    # print(len(traj_test))
    # print(traj_val)


def filter_edge_cases(traj, track): 
    size = np.shape(traj)[0]
    idx = np.zeros((size, 1))

    for k in range(size):
        t = traj[k,2]

        if np.shape(track[traj[k,1]])[1] > 30 and track[traj[k,1]][0, 30] < t and track[traj[k,1]][0,-1] > t+1:
            idx[k] = 1       

    return traj[np.where(idx == 1)[0],:]




def px_to_ft(traj, homography_dir):

    m_to_ft = 3.28084
    h = np.loadtxt(homography_dir, delimiter=' ')
    c_x = 1280/2
    c_y = 720/2

    traj[:,3] = traj[:,3] - c_x
    traj[:,4] = traj[:,4] - c_y
    traj[:,3:5] = multiply_homography(h, traj[:,3:5]) * m_to_ft
    print("Finish converting pixel to feet")
    return traj

def multiply_homography(h, pt_in):
    # a = np.transpose(pt_in)
    # b = np.ones((1, np.shape(pt_in)[0]))
    # print(np.concatenate((a,b)))
    pt = np.matmul(h, np.concatenate((np.transpose(pt_in), np.ones((1, np.shape(pt_in)[0])))))
    pt = np.transpose(pt[0:2,:])
    # print(pt)
    return pt

def tranform(file_dir, homography_dir, out_dir, sgan_dir, toFeet):
    
    read = np.loadtxt(file_dir, delimiter=',')
    traj = np.zeros((np.shape(read)[0], 47))

    traj[:,:5] = read
    ids = np.unique(traj[:,1])

    # print(438 in traj[:,1])
    # print(5220 in traj[:,1])
    # print(3434 in traj[:,1])
    # print(2701 in traj[:,1])

    print("Start importing data...")
    # for k in range(2):
    for k in range(np.shape(traj)[0]):
        # print("Progress: {}/{} ...".format((k+1), np.shape(traj)[0]))
        dsid = traj[k][0]
        vehid = traj[k][1]
        time = traj[k][2]
        vehtraj = traj[traj[:,1] == vehid]      

        # vehtraj = vehtraj[vehtraj[:,0] == dsid]

            # print(np.shape(vehtraj))
        # print(vehtraj)
        frameEgo = traj[traj[:,2] == time]
        
        # frameEgo = frameEgo[frameEgo[:,0] == dsid]
        # print(np.shape(frameEgo))
        # print(frameEgo)
        
        if frameEgo.size != 0:
            dx = np.zeros(np.shape(frameEgo)[0])
            dy = np.zeros(np.shape(frameEgo)[0])
            vid = np.zeros(np.shape(frameEgo)[0])
            
            for l in range(np.shape(frameEgo)[0]):
                dx[l] = frameEgo[l][3] - traj[k][3]
                dy[l] = frameEgo[l][4] - traj[k][4]
                vid[l] = frameEgo[l][1]
            dist = dx*dx + dy*dy
            
            lim = 39

            if len(dist) > lim:
                idx = np.argsort(dist)
                dx = np.array([dx[i] for i in idx[:lim]])
                dy = np.array([dy[i] for i in idx[:lim]])
                vid = np.array([vid[i] for i in idx[:lim]])

            # left
            xl = dx[dx < 0]
            yl = dy[dx < 0]
            vidl = vid[dx < 0]

            yl_top = yl[yl>=0]
            yl_bot = yl[yl<0]
            vidl_top = vidl[yl>=0]
            vidl_bot = vidl[yl<0]

            # center
            xc = dx[dx >= 0]
            yc = dy[dx >= 0]
            vidc = vid[dx >= 0]
            yc = yc[xc < 200]
            vidc = vidc[xc < 200]
            xc = xc[xc < 200]

            yc_top = yc[yc>=0]
            yc_bot = yc[yc<0]
            vidc_top = vidc[yc>=0]
            vidc_bot = vidc[yc<0]

            # right
            xr = dx[dx >= 200]
            yr = dy[dx >= 200]
            vidr = vid[dx >= 200]

            yr_top = yr[yr>=0]
            yr_bot = yr[yr<0]
            vidr_top = vidr[yr>=0]
            vidr_bot = vidr[yr<0]


            # parameters
            mini_top = 7
            mini_bot = 6

            # left top
            iy = np.argsort(yl_top)
            iy = iy[0:min(mini_top, len(yl_top))]
            yl_top = np.array([yl_top[i] for i in iy])
            vidl_top = np.array([vidl_top[i] for i in iy])
            # left bottom
            iy = np.argsort(yl_bot)
            iy = np.array(list(reversed(iy)))
            iy = iy[0:min(mini_bot, len(yl_bot))]
            yl_bot = np.array([yl_bot[i] for i in iy])
            vidl_bot = np.array([vidl_bot[i] for i in iy])

            # center top
            iy = np.argsort(yc_top)
            iy = iy[0:min(mini_top, len(yc_top))]
            yc_top = np.array([yc_top[i] for i in iy])
            vidc_top = np.array([vidc_top[i] for i in iy])
                # center bottom
            iy = np.argsort(yc_bot)
            iy = np.array(list(reversed(iy)))
            iy = iy[0:min(mini_bot, len(yc_bot))]
            yc_bot = np.array([yc_bot[i] for i in iy])
            vidc_bot = np.array([vidc_bot[i] for i in iy])

            # right top
            iy = np.argsort(yr_top)
            iy = iy[0:min(mini_top, len(yr_top))]
            yr_top = np.array([yr_top[i] for i in iy])
            vidr_top = np.array([vidr_top[i] for i in iy])
            # right bottom
            iy = np.argsort(yr_bot)
            iy = np.array(list(reversed(iy)))
            iy = iy[0:min(mini_bot, len(yr_bot))]
            yr_bot = np.array([yr_bot[i] for i in iy])
            vidr_bot = np.array([vidr_bot[i] for i in iy])



            traj[k,8:14] = np.concatenate((np.zeros(6 - len(vidl_bot)),vidl_bot))
            traj[k,14:21] = np.concatenate((vidl_top ,np.zeros(7 - len(vidl_top))))
            traj[k,21:27] = np.concatenate((np.zeros(6 - len(vidc_bot)),vidc_bot))
            traj[k,27:34] = np.concatenate((vidc_top ,np.zeros(7 - len(vidc_top))))
            traj[k,34:40] = np.concatenate((np.zeros(6 - len(vidr_bot)),vidr_bot))
            traj[k,40:47] =np.concatenate((vidr_top ,np.zeros(7 - len(vidr_top))))


    # convert from pixel to feet

    if toFeet:
        traj = px_to_ft(traj, homography_dir)

    # print(438 in traj[:,1])
    # print(5220 in traj[:,1])
    # print(3434 in traj[:,1])
    # print(2701 in traj[:,1])

    # create track
    ids = np.unique(traj[:,1])
    track = {} 
    for i in range(len(ids)):
        vtrack = traj[traj[:,1] == ids[i]]
        track[ids[i]] = vtrack[:,2:5].T 


        
    # filter edge cases
    # print("Filtering edge cases...")
    # traj = filter_edge_cases(traj, track)
    # print("Done filtering edge cases")

    # update tracks according to the change in traj
    # ids = np.unique(traj[:,1])
    # track = {} 
    # for i in range(len(ids)):
    #     vtrack = traj[traj[:,1] == ids[i]]
    #     track[ids[i]] = vtrack[:,2:5].T 

    # print(traj)
    # print(track)


    # f = open(sgan_dir, 'w')
    # for line in traj:
    #     f.write("{}\t{}\t{}\t{}\n".format(int(line[2]), int(line[1]), line[3], line[4]))
    # f.close()


    np.save(out_dir, np.array([traj, track]))
    print("Finish importing and saving")



# # # import_data('../resources/data/TRAF/TRAF11/formatted_hypo.txt', '../resources/data/TRAF/TRAF11/TRAF11_H.txt', '../resources/data/TRAF/TRAF11/TRAF11.npy')

# import_data('../resources/data/TRAF/TRAF12/formatted_hypo.txt', '../resources/data/TRAF/TRAF12/TRAF12_H.txt', '../resources/data/TRAF/TRAF12/TRAF12.npy')
# merge_n_split("../resources/data/TRAF/TRAF{}/TRAF{}.npy", [11, 12], "../Prediction/data/TRAF/TRAF{}.npy")
