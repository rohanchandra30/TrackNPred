# vim: expandtab:ts=4:sw=4
import numpy as np
import os

import cv2
import colorsys
from .image_viewer import ImageViewer
from matplotlib import pyplot as plt
from model.Tracking.deep_sort.detection import Detection
from model.Tracking.application_util import preprocessing




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

def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


def create_class_color(agent_cls):
    if agent_cls == 1: return 0, 0, 255
    elif agent_cls == 2 or agent_cls == 4: return 255,0,255
    elif agent_cls == 3: return 0, 255, 0
    elif agent_cls == 6: return 255, 255, 0
    elif agent_cls == 8: return 0, 0, 255
    else: return 0, 255, 255

class NoVisualization(object):
    """
    A dummy visualization object that loops through all frames in a given
    sequence to update the tracker without performing any visualization.
    """

    def __init__(self, seq_info):
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]

    def set_image(self, image):
        pass

    def draw_groundtruth(self, track_ids, boxes):
        pass

    def draw_detections(self, detections):
        pass

    def draw_trackers(self, trackers):
        pass

    def run(self, frame_callback):
        while self.frame_idx <= self.last_idx:
            frame_callback(self, self.frame_idx)
            self.frame_idx += 1


class Visualization(object):
    """
    This class shows tracking output in an OpenCV image viewer.
    """

    def __init__(self, seq_info, update_ms, loc, thread=None):
        image_shape = seq_info["image_size"][::-1]
        aspect_ratio = float(image_shape[1]) / image_shape[0]
        # image_shape = 1024, 1024
        image_shape = 1024, int(aspect_ratio * 1024)
        self.viewer = ImageViewer(update_ms, image_shape, "Figure %s" % seq_info["sequence_name"])
        self.viewer.thickness = 2
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]
        self.vid_name = seq_info["sequence_name"]
        self.viewer.enable_videowriter(seq_info["sequence_dir"] + "/video/%s.mov" % seq_info["sequence_name"], fourcc_string='XVID', fps=20)
        self.save_dir = loc
        self.thread = thread

    def run(self, frame_callback, thread=None):
        self.viewer.run(lambda: self._update_fun(frame_callback, thread))

    def _update_fun(self, frame_callback, thread):
        if self.frame_idx > self.last_idx:
            return False  # Terminate
        frame_callback(self, self.frame_idx)
        if thread:
            thread.signalTopLabel("Saving results: %05d/%05d" % (self.frame_idx, self.last_idx))
            thread.signalTopBar(max((self.frame_idx / self.last_idx) * 100, 1))
        self.frame_idx += 1
        return True

    # def visualize(self, tracker, seq_info, track_dic, min_confidence,
    #     nms_max_overlap, min_detection_height, thread):
    #     results = []
    #     while self.frame_idx <= self.last_idx:
    #         results = self.process_frame(tracker, results, seq_info, track_dic, min_confidence,
    #     nms_max_overlap, min_detection_height)
    #         if thread:
    #             thread.signalTopLabel("Saving results: %05d/%05d" % (self.frame_idx, self.last_idx))
    #             thread.signalTopBar(max((self.frame_idx / self.last_idx) * 100, 1))
    #         self.frame_idx += 1
    #     return results


    # def process_frame(self, tracker, results, seq_info, track_dic, min_confidence,
    #     nms_max_overlap, min_detection_height):


    #     detections = create_detections(seq_info["detections"], self.frame_idx, min_detection_height)
    #     detections = [d for d in detections if d.confidence >= min_confidence]

    #     # Run non-maxima suppression.
    #     boxes = np.array([d.tlwh for d in detections])
    #     scores = np.array([d.confidence for d in detections])
    #     indices = preprocessing.non_max_suppression(
    #         boxes, nms_max_overlap, scores)
    #     detections = [detections[i] for i in indices]

    #     # Update tracker.
    #     tracker.predict()
    #     tracker.update(detections)

    #     # Update visualization.
    #     image = cv2.imread(seq_info["image_filenames"][self.frame_idx], cv2.IMREAD_COLOR)
    #     # try:
    #     r = image.shape[0]
    #     c = image.shape[1]
    #     shp = (r, c, 1)
    #     c_red, c_green, c_blue = cv2.split(image)
    #     alpha = 0.5
    #     alphachn = np.ones(shp, dtype=np.uint8) * int(alpha * 255)
    #     image = cv2.merge((c_red, c_green, c_blue, alphachn))
    #     self.set_image(image.copy())
    #     self.draw_trackers(track_dic, self.frame_idx,tracker.tracks)
    #     # except:
    #     #     print('go on')

    #     # Store results.
    #     # n_tracks = len(tracker.tracks)
    #     for track in tracker.tracks:
    #         # if not track.is_confirmed() or track.time_since_update > 1:
    #         if not track.is_confirmed():
    #             continue
    #         bbox = track.to_tlwh()
    #         results.append([
    #             self.frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    #     return results

    def set_image(self, image):
        self.viewer.image = image
        # self.viewer.image = np.zeros(image.shape(), np.uint8)

    def draw_groundtruth(self, track_ids, boxes):
        self.viewer.thickness = 2
        for track_id, box in zip(track_ids, boxes):
            self.viewer.color = create_unique_color_uchar(track_id)
            self.viewer.rectangle(*box.astype(np.int), label=str(track_id))

    def draw_detections(self, detections):
        self.viewer.thickness = 2
        self.viewer.color = 0, 0, 255
        for i, detection in enumerate(detections):
            # print(detection.tlwh)
            self.viewer.rectangle(*detection.tlwh)
            self.viewer.circle(detection.tlwh[0]+detection.tlwh[2], detection.tlwh[1]+detection.tlwh[3], 2)


    def draw_trackers(self, track_dic, idx,tracks):
        stride = 3
        self.viewer.thickness = 3
        # print (self.vid_name, idx)
        for track in tracks:
            track_dic.setdefault(track.track_id, []).append(list(list(track.to_tlwh().astype(np.int))))
            # if not track.is_confirmed() or track.time_since_update > 0:
            #     continue
            # print(track.track_id)
            # self.viewer.color = create_class_color(track.agent_cls)
            self.viewer.color = create_unique_color_uchar(track.track_id)
            # print(self.viewer.color)
            self.viewer.rectangle(idx, self.vid_name, *track.to_tlwh().astype(np.int), label=str(track.track_id), loc=self.save_dir, thread=self.thread)
            # if len(track_dic[track.track_id]) == 1:
            # self.viewer.rectangle(idx, self.vid_name, track_dic[track.track_id][i][0],
            #                           track_dic[track.track_id][i][1], track_dic[track.track_id][i][2],
            #                           track_dic[track.track_id][i][3], label=str(track.track_id))

            # Moving average
            # N = len(track_dic[track.track_id])
            # x = track_dic[track.track_id]
            # track_x = []
            # track_y = []
            # for k in range(N):
            #     track_x.append(x[k][0])
            #     track_y.append(x[k][1])

            # for i in range(len(track_dic[track.track_id])):
            #     idx_curr = int(i/stride)*stride
            #     if len(track_dic[track.track_id]) > stride and idx_curr < (len(track_dic[track.track_id]) - stride):
            #         self.viewer.line(idx, track.agent_cls, self.vid_name,
            #                          track_dic[track.track_id][idx_curr][0], track_dic[track.track_id][idx_curr][1],
            #                          track_dic[track.track_id][idx_curr][2], track_dic[track.track_id][idx_curr][3],
            #                          track_dic[track.track_id][idx_curr+stride][0], track_dic[track.track_id][idx_curr+stride][1],
            #                          track_dic[track.track_id][idx_curr+stride][2], track_dic[track.track_id][idx_curr+stride][3],
            #                          *track.to_tlwh().astype(np.int), label=str(track.track_id))
            # self.viewer.marker(idx, track.agent_cls, self.vid_name, *track.to_tlwh().astype(np.int), label=str(track.track_id))
            # self.viewer.line(idx, track.agent_cls, self.vid_name,
            #                  np.mean(track_x[:idx_curr]), np.mean(track_y[:idx_curr]),
            #                  track_dic[track.track_id][idx_curr][2], track_dic[track.track_id][idx_curr][3],
            #                  np.mean(track_x[:(idx_curr+stride)]), np.mean(track_y[:(idx_curr+stride)]),
            #                  track_dic[track.track_id][idx_curr+stride][2], track_dic[track.track_id][idx_curr+stride][3],
            #                  *track.to_tlwh().astype(np.int), label=str(track.track_id))

            # self.viewer.circle(idx,self.vid_name,int((track_dic[track.track_id][i][0]+track_dic[track.track_id][i][2])/2),int((track_dic[track.track_id][i][1]+track_dic[track.track_id][i][3])/2),4, label=str(track.track_id))
        # print type(track_dic[2])
        # return track_dic

    def draw_trackers_in_agent(self, track_dic, idx, tracks):
        stride = 1
        agent_count = 0
        self.viewer.thickness = 3
        # print (self.vid_name, idx)
        for track in tracks:
            track_dic.setdefault(track.track_id, []).append(list(list(track.to_tlwh().astype(np.int))))
            # if track_dic[track.track_id] is None:
            #     track_dic[track.track_id].append(list(list(track.to_tlwh().astype(np.int))))
            # else:
            #     track_dic[track.track_id] = []
            #     track_dic[track.track_id].append(list(list(track.to_tlwh().astype(np.int))))
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            self.viewer.color = create_class_color(track.agent_cls)
            # self.viewer.color = create_unique_color_uchar(track.agent_cls)
            # self.viewer.rectangle(idx, self.vid_name, *track.to_tlwh().astype(np.int), label=str(track.track_id))
            self.viewer.marker_in_cls(idx, agent_count, self.vid_name, *track.to_tlwh().astype(np.int), label=str(track.track_id))
            # if len(track_dic[track.track_id]) == 1:
            #     self.viewer.rectangle(idx, self.vid_name, track_dic[track.track_id][i][0],
            #                           track_dic[track.track_id][i][1], track_dic[track.track_id][i][2],
            #                           track_dic[track.track_id][i][3], label=str(track.track_id))
            for i in range(len(track_dic[track.track_id])):
                idx_curr = int(i/stride)*stride
                if len(track_dic[track.track_id]) > stride and i < (len(track_dic[track.track_id]) - stride) and track.agent_cls == 1.0:
                    self.viewer.line_in_cls(idx, agent_count, self.vid_name,
                                     track_dic[track.track_id][idx_curr][0], track_dic[track.track_id][idx_curr][1],
                                     track_dic[track.track_id][idx_curr][2], track_dic[track.track_id][idx_curr][3],
                                     track_dic[track.track_id][idx_curr+stride][0], track_dic[track.track_id][idx_curr+stride][1],
                                     track_dic[track.track_id][idx_curr+stride][2], track_dic[track.track_id][idx_curr+stride][3],
                                     *track.to_tlwh().astype(np.int), label=str(track.track_id))
            agent_count+=1


    def draw_traj(self, track_dic, track):
        for i in range(len(track_dic[track.track_id])):
            if len(track_dic[track.track_id]) > 1 and i != (len(track_dic[track.track_id]) - 1):
                self.viewer.line(idx, self.vid_name, track_dic[track.track_id][i][0],
                                 track_dic[track.track_id][i][1], track_dic[track.track_id][i][2],
                                 track_dic[track.track_id][i][3], track_dic[track.track_id][i + 1][0],
                                 track_dic[track.track_id][i + 1][1], track_dic[track.track_id][i + 1][2],
                                 track_dic[track.track_id][i + 1][3], *track.to_tlwh().astype(np.int),
                                 label=str(track.track_id))
