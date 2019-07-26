import cv2
import datetime as dt
import matplotlib.image as mpimg
import model.Detection.Mask.mrcnn.model as modellib
import numpy as np
import os
import sys
import time

from model.Detection.Mask.mrcnn import utils
from model.Detection.Mask.mrcnn import visualize
from resources.mask_resources.coco import coco

dataset = 'TRAF_Dataset'
all_videos = ['CroppedTRAF12.mp4']


def extract_frames(video_path, framesDir):
    print("FRMEAS DIR: ", framesDir)
    print("vid path: ", video_path)
    frames = sorted(os.listdir(framesDir))
    if len(frames) == 0:
        video_object = cv2.VideoCapture(os.path.join(video_path))
        count = 0
        while video_object.isOpened():
            success, _curr_frame = video_object.read()
            if not success:
                print("could not open video object")
                break
            cv2.imwrite(os.path.join(framesDir, '{:06}.jpg'.format(count)), _curr_frame)
            count += 1
        video_object.release()
        frames = sorted(os.listdir(framesDir))
    return frames


def find_labels(_class_names, items):
    return [_class_names.index(i) for _, i in enumerate(items)]


def save_detections(_fidx, frame, mrcnn_results, _detection_file, inputDir, det_frames_dir, _labels_to_save=None, thread=None):
    print(_detection_file)
    file_id = open(_detection_file, 'a+')
    class_ids = mrcnn_results['class_ids']
    if _labels_to_save is None:
        labels_idx = [idx for idx, i in enumerate(class_ids)]
    else:
        labels_idx = [idx for idx, i in enumerate(class_ids) if i in _labels_to_save]
    rois = mrcnn_results['rois'][labels_idx, :]
    masks = mrcnn_results['masks'][:, :, labels_idx]
    scores = mrcnn_results['scores'][labels_idx]

    for i in range(0, len(rois)):
        mask = masks[:, :, i]
        seg = rois[i]
        seg[:2] = np.maximum(0, seg[:2])
        seg[2:] = np.minimum(np.asarray(visualize.apply_mask(frame, mask, (1, 1, 1), alpha=0.5).
                                        shape[:2][::-1]) - 1, seg[2:])
        if np.any(seg[:2] >= seg[2:]):
            continue
        delimiter = ','
        # line = (str(_fidx), "-1", str(seg[1]), str(seg[0]), str(seg[3] - seg[1]), str(seg[2] - seg[0]), str(scores[i]), "-1", "-1", "-1", str(class_ids[i]))
        file_id.write("{},-1,{},{},{},{},{},-1,-1,-1,{}\n".format(str(_fidx), str(seg[1]), str(seg[0]), str(seg[3] - seg[1]), str(seg[2] - seg[0]), str(scores[i]), str(class_ids[i])))


        print("det dir: ", det_frames_dir)

        det_frame_path = os.path.join(inputDir, det_frames_dir, '{:06}.jpg'.format(_fidx))
        print(det_frame_path)
        cv2.imwrite(det_frame_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if thread:
            thread.signalImg(det_frame_path)

    file_id.close()


# Main Code

def detect(inputDir, inputFile, framesDir, outputPath, outputFolder, conf, nms, cuda, thread=None):

    # Source code directory of the project
    SRC_DIR = "resources"

    # Data directory for current dataset
    # DATA_DIR = os.path.abspath(os.path.join(SRC_DIR, '../../croncal/data/{}'.format(dataset)))

    # Import Mask RCNN
    # sys.path.append(SRC_DIR)  # To find local version of the library

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(SRC_DIR, 'mask_resources')

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(MODEL_DIR, 'mask_rcnn_coco.h5')
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        utils.download_trained_weights(COCO_MODEL_PATH)


    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.7


    config = InferenceConfig()
    # config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = [
        # 1                 # 2                 # 3             # 4                 # 5
        'BG',               'person',           'bicycle',      'car',              'motorcycle',       # 1
        'airplane',         'bus',              'train',        'truck',            'boat',             # 2
        'traffic light',    'fire hydrant',     'stop sign',    'parking meter',    'bench',            # 3
        'bird',             'cat',              'dog',          'horse',            'sheep',            # 4
        'cow',              'elephant',         'bear',         'zebra',            'giraffe',          # 5
        'backpack',         'umbrella',         'handbag',      'tie',              'suitcase',         # 6
        'frisbee',          'skis',             'snowboard',    'sports ball',      'kite',             # 7
        'baseball bat',     'baseball glove',   'skateboard',   'surfboard',        'tennis racket',    # 8
        'bottle',           'wine glass',       'cup',          'fork',             'knife',            # 9
        'spoon',            'bowl',             'banana',       'apple',            'sandwich',         # 10
        'orange',           'broccoli',         'carrot',       'hot dog',          'pizza',            # 11
        'donut',            'cake',             'chair',        'couch',            'potted plant',     # 12
        'bed',              'dining table',     'toilet',       'tv',               'laptop',           # 13
        'mouse',            'remote',           'keyboard',     'cell phone',       'microwave',        # 14
        'oven',             'toaster',          'sink',         'refrigerator',     'book',             # 15
        'toothbrush'                                                                                    # 16
    ]

    video_path = os.path.join(inputDir, inputFile)

    # directory for normal frames
    # framesDir = os.path.join(inputDir, framesDir)
    # os.makedirs(framesDir, exist_ok=True)

    # det,txt path
    detection_file = os.path.join(inputDir, outputPath)

    # directory for mask rcnn detection frames
    os.makedirs(os.path.join(inputDir, outputFolder), exist_ok=True)

    framesPath = os.path.join( inputDir, framesDir)
    os.makedirs(framesPath, exist_ok=True)
    # checks if there are enough frames in framesPath

    num_frames = len(os.listdir(framesPath))
    print(framesPath, num_frames)
    if num_frames < 500:
        if(thread):
            thread.signalCanvas("Extracting frames from {}".format(inputFile))
        frames = extract_frames(video_path, framesDir)
    else:
        if(thread):
            thread.signalCanvas("Found {} frames in  {}. Delete this folder to re-extract frames".format(num_frames, framesPath))
        frames = sorted(os.listdir(framesPath))
    if(thread):
        thread.signalCanvas("Extracting frames from {}".format(inputFile))

    start_time_each_video = time.time()
    time_per_frame = []
    
    if thread:
        thread.signalCanvas("\nPerforming object detection:")
        thread.signalTopLabel("{}: 0/{} batches".format(inputFile, num_frames))


    for fidx, each_frame in enumerate(frames):

        if(thread):
            tlbl = "{}: {}/{} frames".format(inputFile, fidx + 1, num_frames)
            thread.signalTopLabel(tlbl)
            thread.signalTopBar(max( (((fidx + 1) / num_frames) * 100), 1))

        curr_frame = np.array(mpimg.imread(os.path.join(framesPath, each_frame)))

        start_time_each_frame = time.time()
        
        results = model.detect([curr_frame], verbose=0)[0]
        labels_to_save = find_labels(class_names, ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'])
        
        # Save detection results to det.txt file
        if labels_to_save is None:
            save_detections(fidx, curr_frame, results, detection_file, inputDir, outputFolder, labels_to_save, thead)
            
            time_per_frame.append(time.time() - start_time_each_frame)
            mean_time_per_frame = np.mean(np.array(time_per_frame))
            ttc = round((num_frames-fidx) * mean_time_per_frame)
            
            rois = results['rois']
            masks = results['masks']
            labels = results['class_ids']
        else:
            save_detections(fidx, curr_frame, results, detection_file, inputDir, outputFolder, labels_to_save, thread)
            
            # Create mask to remove detections whose labels are not in labels_to_save
            results_labels_mask = [False] * len(results['class_ids'])
            for idx, label in enumerate(results['class_ids']):
                if label in labels_to_save:
                    results_labels_mask[idx] = True
            
            time_per_frame.append(time.time() - start_time_each_frame)
            mean_time_per_frame = np.mean(np.array(time_per_frame))
            ttc = round((num_frames-fidx) * mean_time_per_frame)
            rois = results['rois'][results_labels_mask]
            masks = results['masks'][:, :, results_labels_mask]
            labels = results['class_ids'][results_labels_mask]
            
            # print(rois.shape)
            # print(masks.shape)
            # print(labels.shape)
            
            # visualize.display_instances(curr_frame, rois, masks, labels, np.array(class_names))
      
            print('\rVideo: {}. Processing: {:.2f}%. TTC: {} (mean time per frame: {:.2f} secs).'.
                  format(inputFile, fidx * 100. / num_frames,
                         str(dt.timedelta(seconds=ttc)), mean_time_per_frame), end='')

    print('\rVideo: {}. Processing: done. Total time taken: {}.'.
          format(inputFile, str(dt.timedelta(seconds=round(time.time() - start_time_each_video)))))

        
    
