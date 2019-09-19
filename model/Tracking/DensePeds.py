from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np
import time
from model.Tracking.application_util import preprocessing
from model.Tracking.application_util import visualization
from model.Tracking.deep_sort import nn_matching
from model.Tracking.deep_sort.detection import Detection
from model.Tracking.deep_sort.tracker import Tracker

# dataset = 'resources/data/TRAF'
# dataset = './Tracking'
# video_folder = ['TRAF11']
track_dic = {}

def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameterskf
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    # print(sequence_dir)
    # print(os.path.join(sequence_dir, "frames"))
    image_dir = os.path.join(sequence_dir, "frames")
    # image_dir = "../data/TRAF12/frames"
    image_filenames = {
        # int(os.path.splitext(f)[0].split('_')[1]): os.path.join(image_dir, f)
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        # print(detections)
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None
    if detections.ndim == 1 and detections is not None:
        feature_dim = len(detections) - 10
    elif detections.ndim > 1 and detections is not None:
        feature_dim = detections.shape[1] - 10
    else:
        feature_dim = 0
    # feature_dim = detections.shape[1] - 10 if detections is not None and detections.ndim>1 else 0
    seq_info = {
        "sequence_dir": sequence_dir,
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int) if detection_mat.ndim>1 else detection_mat[0]
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, agent_cls, feature = row[2:6], row[6], row[10], row[11:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, agent_cls, feature))
    return detection_list


def run(sequence_dir, detection_file, output_file,matlab_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display, thread):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    def frame_callback(vis, frame_idx):
        # Load image and generate detections.
        detections = create_detections(seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
            image = cv2.imread(seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            r = image.shape[0]
            c = image.shape[1]
            shp = (r, c, 1)
            c_red, c_green, c_blue = cv2.split(image)
            alpha = 0.5
            alphachn = np.ones(shp, dtype=np.uint8) * int(alpha * 255)
            image = cv2.merge((c_red, c_green, c_blue, alphachn))
            vis.set_image(image.copy())
            vis.draw_trackers(track_dic, frame_idx, tracker.tracks)
            
        # Store results.
        # n_tracks = len(tracker.tracks)
        for track in tracker.tracks:
            # if not track.is_confirmed() or track.time_since_update > 1:
            if not track.is_confirmed():
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if display:
        save_loc = os.path.join(sequence_dir,'trackingFrames')
        print(save_loc)
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)
        visualizer = visualization.Visualization(seq_info, update_ms=5, loc=save_loc, thread=thread)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    if thread:
        visualizer.run(frame_callback, thread)
    else:
        visualizer.run(frame_callback)

    # if thread:
    #     thread.signalTopLabel("Saving results: %05d/%05d" % (0, seq_info["max_frame_idx"]))
    #     thread.signalTopBar(0)

    # results = visualizer.visualize(tracker, seq_info, track_dic, min_confidence,
    #     nms_max_overlap, min_detection_height, thread)

    # Store results.
    f = open(output_file, 'w+')
    # print(results)
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=False)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=False)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.2, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=100)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool)
    parser.add_argument("--video_folder", '-v', required=True)
    return parser.parse_args()

def densepeds(dataDir, video_folder, display=True, thread=None):
    min_conf = 0.2
    nms = 1.0
    min_det_ht = 0
    max_cos_dist = 0.2
    nn_budget = 100
    disp = display

    dataset = dataDir

    # for video_file in [video_folder]:

    track_dic = {}

    print('DATASET densepeds:', dataset)

    start_time = time.time()
    sequence_dir = dataset + '/'+ video_folder
    detection_file = dataset + '/'+  video_folder + '/densep.npy'
    output_file = dataset + '/' +  video_folder + '/hypotheses.txt'
    matlab_file= '../MOT_Dataset/' +'amilan-motchallenge-devkit-7dccd0fb3214/res/MOT16/data/rohan/train_87.5conf/' + video_folder + '.txt'


    run(sequence_dir, detection_file, output_file, matlab_file, min_conf, nms, min_det_ht, max_cos_dist, nn_budget, disp, thread)
    print(video_folder, (time.time() - start_time))


# if __name__ == "__main__":
#     args = parse_args()
#     for video_file in [args.video_folder]:
#         print(video_file)
#         track_dic = {}
#         # print("69" * 69, "listdir: ", os.listdir("../data/TRAF12/frames"))

#         start_time = time.time()
#         print(dataset)
#         sequence_dir = dataset + '/'+ video_file
#         detection_file = dataset + '/'+  video_file + '/densep.npy'
#         # print(detection_file)
#         output_file = dataset + '/' +  video_file + '/hypotheses.txt'
#         matlab_file= '../MOT_Dataset/' +'amilan-motchallenge-devkit-7dccd0fb3214/res/MOT16/data/rohan/train_87.5conf/' + video_file + '.txt'
#         # run(sequence_dir, detection_file, output_file, matlab_file,args.min_confidence, args.nms_max_overlap, args.min_detection_height, args.max_cosine_distance, args.nn_budget, args.display)
#         # print(video_file, (time.time() - start_time))

