
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.io
import os
import numpy as np
import random
import re

video_folder = "1FWEST/"
dataset = 'Alibaba_Dataset/'

def display_instances(image, boxes,nn, class_ids='label', title="", figsize=(16, 16), ax=None):

    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

        # Show area outside image boundaries.
        height, width = image.shape[:2]
        ax.set_ylim(height + 10, -10)
        ax.set_xlim(-10, width + 10)
        ax.axis('off')
        ax.set_title(title)

    color='red'

    for box in boxes:

        masked_image = image.astype(np.uint32).copy()
        # Bounding box
        xy = re.split('[,\n]', box)
        x1=int(xy[0])
        y1=int(xy[1])
        x2=int(xy[0])+int(xy[2])
        y2 =int (xy[1]) + int(xy[3])
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7,
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)


        # Label
        label = int(xy[4])
        x = random.randint(x1, (x1 + x2) // 2)
        caption = label
        ax.text(x1, y1 + 8, caption, color='w', size=11, backgroundcolor="none")

#plt.ion()
    ax.imshow(masked_image.astype(np.uint8))
    savefile='../Alibaba_Dataset/saveimg/'+str(nn)+'.jpg'
    plt.savefig(savefile)
#     plt.show()
#     plt.pause(0.1)
    plt.close()


img_dir    ='../Alibaba_Dataset/saveBlurImg1/saveBlurImg1'
gt_dir     ='../Alibaba_Dataset/gtAll/gtAll'
file_names = os.listdir(img_dir)

ax = None
frame=0
output_file = '../' + dataset  + video_folder + 'gt/gt.txt'
for all_dir in file_names[2000:]:
    print(all_dir)
    image  = skimage.io.imread(os.path.join(img_dir,all_dir))
    name   = ("%05d" % int(all_dir.split('.')[0])) + '.txt'
    file_p = open(os.path.join(gt_dir,name))
    boxes  = file_p.readlines()
    # f = open(output_file, 'a')
    # for box in boxes:
    #     xy = re.split('[,\n]', box)
    #     x1 = int(xy[0])
    #     y1 = int(xy[1])
    #     w = int(xy[2])
    #     h = int(xy[3])
    #     id = int(xy[4])
    ax     = display_instances(image, boxes,frame)
    frame=frame+1
    #     print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, id, x1, y1, w, h), file=f)