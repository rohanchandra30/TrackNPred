import os
import numpy as np
from collections import Counter
from itertools import dropwhile

from matplotlib import pyplot as plt
import matplotlib.image as mpimg

video_folder = 'IITF-1'
dataset = 'Aniket_Dataset'


def process(line):
    out_file = open('/scratch1/Research/Aniket_Dataset/%s/gt/gt.txt' % (video_folder), 'a')
    my_str = ","
    seq = (str(int(line[1])), str(int(line[0])), str(int(line[2])), str(int(line[3])))
    out_file.write(my_str.join(seq))
    out_file.write('\n')
    out_file.close()


def convert(hypo_in, ab_in):
    for i in range(int(hypo_in[0][0]), int(hypo_in[-1][0])):
        for line in ab_in:
            if line[1] == i:
                process(line)

def bbox_has_point(h_row, gt_file, frame):
    # bbox = plt.Rectangle((h_row[2], h_row[3]), h_row[4], h_row[5], fill=False, edgecolor='red', linewidth=3.5)
    for i in range(0, len(gt_file)):
        if gt_file[i][0] == frame:
            if h_row[2] < gt_file[i][2] < h_row[2]+h_row[4] and h_row[3] < gt_file[i][3]<h_row[3]+h_row[5]:
                return True
    return False

def get_current_ID(h_row, gt_file, frame):
    for i in range(0, len(gt_file)):
        if gt_file[i][0] == frame:
            if h_row[2] < gt_file[i][2] < h_row[2]+h_row[4] and h_row[3] < gt_file[i][3]<h_row[3]+h_row[5]:
                return gt_file[i][1], h_row[1]
    return 0,0

def get_previous_ID(h_file, frame, x, y,u,v):
    # for i in range(0, len(gt_file)):
    #     if gt_file[i][0] == frame-1:
    #         if gt_file[i][1] == curr_gid:
    #             x,y = gt_file[i][2], gt_file[i][3]
    for i in range(0, len(h_file)):
        if h_file[i][0] == frame - 1:
            if np.abs(h_file[i][2] - x) <=5 or np.abs(h_file[i][3] - y) <=5 or np.abs(h_file[i][2]+h_file[i][4] - u) <=5 or np.abs(h_file[i][3]+h_file[i][5] - v) <=5:
                return h_file[i][1]

    return 0


def compute_GT(gt_file, frame):
    count = 0
    for i in range(0, len(gt_file)):
        if gt_file[i][0] == frame:
            count = count + 1
    return count

def compute_HT(h_file, frame):
    count = 0
    for i in range(0, len(h_file)):
        if h_file[i][0] == frame:
            count = count + 1
    return count

def compute_FP(h_file, gt_file, frame):
    # count = 0
    # for i in range(0, len(h_file)):
    #     if h_file[i][0] == frame:
    #         if bbox_has_point(h_file[i], gt_file, frame) is False:
    #             count = count + 1
    # return count

    gt_count = compute_GT(gt_file, frame)
    # fp_count = compute_FP(h_file, gt_file, frame)
    h_count = compute_HT(h_file, frame)

    if gt_count <= h_count:
        return h_count - (gt_count)
    else:
        return 0

def compute_FN(h_file, gt_file, frame):
    gt_count = compute_GT(gt_file, frame)
    # fp_count = compute_FP(h_file, gt_file, frame)
    h_count = compute_HT(h_file, frame)

    if gt_count >= h_count:
        return gt_count - (h_count)
    else:
        return 0
def compute_IDSW(h_file, frame):
    count = 0
    if frame == 1:
        return 0
    else:
        for i in range(0, len(h_file)):
            if h_file[i][0] == frame:
                # curr_gid, curr_hid = get_current_ID(h_file[i], gt_file, frame)
                curr_hid,x,y = h_file[i][1],h_file[i][2],h_file[i][3]
                prev_hid = get_previous_ID(h_file, frame, x, y, x+h_file[i][4], y+h_file[i][5])
                if prev_hid != 0 and curr_hid != prev_hid:
                    count = count + 1

    return count

def compute_timespan(h_file):
    counted_list =  Counter(h_file[:,1])
    for key, count in dropwhile(lambda key_count: key_count[1] >= 155, counted_list.most_common()):
        del counted_list[key]
    # if h_file[i][0] == frame:
    #     track_id = 1
    #     for i in range(0, len(h_file)):
    #         if h_file[i][0] == track_id:
    #                 count = count + 1

    return sorted(counted_list.items())

def main():
    # video_folder = 'IITF-1'
    ab_path = os.path.join('/scratch1/Research/Aniket_Dataset', video_folder, 'gt', video_folder+'.txt')
    hypo_path = os.path.join('/scratch1/Research/Aniket_Dataset', video_folder, 'results', 'hypotheses.txt')
    ab_in = np.loadtxt(ab_path, delimiter=' ')
    hypo_in = np.loadtxt(hypo_path, delimiter=',')

    # Convert the .ab file to gt.txt file (prepare it for evaluation)
    convert(hypo_in, ab_in)


    frame = 3
    #img = mpimg.imread('/scratch1/Research/Aniket_Dataset/IITF-1/img1/output_0003.png')
    #fig, ax = plt.subplots(figsize=(12, 12))
    #plt.imshow(img)
    gt = np.loadtxt('/scratch1/Research/Aniket_Dataset/IITF-1/gt/gt.txt', delimiter=',')
    #for i in range(0, len(gt)):
    #    if gt[i][0]==frame:
     #       plt.plot(gt[i][2],gt[i][3], color='green', marker='o', markersize=4)

   # for i in range(1, len(hypo_in)):
    #    if hypo_in[i][0]==frame:
     #       h_row = hypo_in[i]
            # plt.plot(hypo_in[i][2],hypo_in[i][3], color='blue', marker='o', markersize=4)
      #      ax.add_patch(plt.Rectangle((h_row[2], h_row[3]), h_row[4], h_row[5], fill=False, edgecolor='red', linewidth=3.5))
    #plt.draw()
    #plt.show()
    #plt.savefig('/home/rohan/Downloads/box_point.png', bbox_inches='tight')

    #     Evaluation
    gt_count = 0
    fp_count = 0
    fn_count = 0
    idsw_count = 0
    timespan = 0
    timespan = compute_timespan(hypo_in)
    # for i in range(1,len(hypo_in+1)):
       # gt_count = gt_count+compute_GT(gt, i)
       # fp_count = fp_count+compute_FP(hypo_in, gt, i)
       # fn_count = fn_count+compute_FN(hypo_in, gt, i)
       # idsw_count = idsw_count+compute_IDSW(hypo_in, i)
       # timespan = compute_timespan(hypo_in, i)
    # print(1-(idsw_count)/gt_count,idsw_count)
    print(timespan)
    print(len(timespan))

if __name__ == "__main__":
    main()
