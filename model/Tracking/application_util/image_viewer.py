# vim: expandtab:ts=4:sw=4
"""
This module contains an image viewer and drawing routines based on OpenCV.
"""
import numpy as np
import cv2
import time
import os

def is_in_bounds(mat, roi):
    """Check if ROI is fully contained in the image.

    Parameters
    ----------
    mat : ndarray
        An ndarray of ndim>=2.
    roi : (int, int, int, int)
        Region of interest (x, y, width, height) where (x, y) is the top-left
        corner.

    Returns
    -------
    bool
        Returns true if the ROI is contain in mat.

    """
    if roi[0] < 0 or roi[0] + roi[2] >= mat.shape[1]:
        return False
    if roi[1] < 0 or roi[1] + roi[3] >= mat.shape[0]:
        return False
    return True


def view_roi(mat, roi):
    """Get sub-array.

    The ROI must be valid, i.e., fully contained in the image.

    Parameters
    ----------
    mat : ndarray
        An ndarray of ndim=2 or ndim=3.
    roi : (int, int, int, int)
        Region of interest (x, y, width, height) where (x, y) is the top-left
        corner.

    Returns
    -------
    ndarray
        A view of the roi.

    """
    sx, ex = roi[0], roi[0] + roi[2]
    sy, ey = roi[1], roi[1] + roi[3]
    if mat.ndim == 2:
        return mat[sy:ey, sx:ex]
    else:
        return mat[sy:ey, sx:ex, :]


class ImageViewer(object):
    """An image viewer with drawing routines and video capture capabilities.

    Key Bindings:

    * 'SPACE' : pause
    * 'ESC' : quit

    Parameters
    ----------
    update_ms : int
        Number of milliseconds between frames (1000 / frames per second).
    window_shape : (int, int)
        Shape of the window (width, height).
    caption : Optional[str]
        Title of the window.

    Attributes
    ----------
    image : ndarray
        Color image of shape (height, width, 3). You may directly manipulate
        this image to change the view. Otherwise, you may call any of the
        drawing routines of this class. Internally, the image is treated as
        beeing in BGR color space.

        Note that the image is resized to the the image viewers window_shape
        just prior to visualization. Therefore, you may pass differently sized
        images and call drawing routines with the appropriate, original point
        coordinates.
    color : (int, int, int)
        Current BGR color code that applies to all drawing routines.
        Values are in range [0-255].
    text_color : (int, int, int)
        Current BGR text color code that applies to all text rendering
        routines. Values are in range [0-255].
    thickness : int
        Stroke width in pixels that applies to all drawing routines.

    """

    def __init__(self, update_ms, window_shape=(640, 480), caption="Figure 1"):
        self._window_shape = window_shape
        self._caption = caption
        self._update_ms = update_ms
        self._video_writer = None
        self._user_fun = lambda: None
        self._terminate = False

        self.image = np.zeros(self._window_shape + (3, ), dtype=np.uint8)
        self._color = (0, 0, 0)
        self.text_color = (255, 255, 255)
        self.thickness = 1

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        if len(value) != 3:
            raise ValueError("color must be tuple of 3")
        self._color = tuple(int(c) for c in value)

    def rectangle(self,idx, video, x, y, w, h, label=None, loc=None, thread=None):
        """Draw a rectangle.

        Parameters
        ----------
        x : float | int
            Top left corner of the rectangle (x-axis).
        y : float | int
            Top let corner of the rectangle (y-axis).
        w : float | int
            Width of the rectangle.
        h : float | int
            Height of the rectangle.
        label : Optional[str]
            A text label that is placed at the top left corner of the
            rectangle.

        """
        pt1 = int(x), int(y)
        pt2 = int(x + w), int(y + h)
        center = (int(x)+int(x + w))/2, (int(y)+int(y + h))/2
        if self._color == (128,0,0):
            overlay = self.image.copy()
            # cv2.rectangle(overlay, pt1, pt2, self._color, self.thickness)
            alpha = 0.5
            # cv2.addWeighted(overlay, alpha, self.image, 1-alpha, 0, self.image)
        else:
            cv2.rectangle(self.image, pt1, pt2, self._color, self.thickness)
        # cv2.circle(self.image, center, 4, self._color, self.thickness)
        if label is not None:
            text_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_PLAIN, 1, self.thickness)

            center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
            pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + \
                text_size[0][1]
            cv2.rectangle(self.image, pt1, pt2, self._color, self.thickness)
            cv2.putText(self.image, label, center, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(loc, "{0:06d}.jpg".format(idx)), self.image)
        # cv2.imwrite("{0:06d}.jpg".format(idx), self.image)
       # cv2.imwrite("/home/uttaran/Codes/Gamma/DensePeds/Aniket_Dataset/NDLS-2/video/%d.jpg" % (idx), self.image)
        if thread and thread.args["display"]:
            thread.signalImg(os.path.join(loc, "{0:06d}.jpg".format(idx)))


    def marker(self,idx, agent_cls, video, x, y, w, h, label=None):
        """Draw a rectangle.

        Parameters
        ----------
        x : float | int
            Top left corner of the rectangle (x-axis).
        y : float | int
            Top let corner of the rectangle (y-axis).
        w : float | int
            Width of the rectangle.
        h : float | int
            Height of the rectangle.
        label : Optional[str]
            A text label that is placed at the top left corner of the
            rectangle.

        """
        if agent_cls == 1 or agent_cls == 2 or agent_cls == 4:
            center = int((2*x+w) / 2)-6, int(y)+6
        elif agent_cls == 3:
            center = int((2*x+w) / 2), int((2*y+h) / 2)
        else:
            center = int((2*x+w) / 2), int(y)
        radius = 13
        center_txt = int((2*x+w) / 2)-14, int(y)+14
        cv2.circle(self.image, center, radius, (0, 255, 255), -1)
        # cv2.circle(self.image, center, 4, self._color, self.thickness)
        if label is not None:
            # text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, self.thickness)

            # center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
            # pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + \
            #     text_size[0][1]
            # cv2.circle(self.image, center, radius, (0, 255, 255), -1)
            cv2.putText(self.image, label, center_txt, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        # cv2.imwrite("../Traffic_Dataset/%s/video/%d.png" % (video,idx), self.image)

    def marker_in_cls(self,idx, agent_count, video, x, y, w, h, label=None):
        """Draw a rectangle.

        Parameters
        ----------
        x : float | int
            Top left corner of the rectangle (x-axis).
        y : float | int
            Top let corner of the rectangle (y-axis).
        w : float | int
            Width of the rectangle.
        h : float | int
            Height of the rectangle.
        label : Optional[str]
            A text label that is placed at the top left corner of the
            rectangle.

        """
        if agent_count % 5 == 0:
            center = int((2*x + w) / 2), int(y + h)
        if agent_count % 5 == 1:
            center = int((2*x + w) / 2), int((4*y + h) / 4)
        elif agent_count % 5 == 2:
            center = int((2*x + w) / 2), int((2*y + h) / 2)
        if agent_count % 5 == 3:
            center = int((2*x + w) / 2), int((y + 4*h) / 4)
        elif agent_count % 5 == 4:
            center = int((2*x + w) / 2), int(y)
        radius = 7
        cv2.circle(self.image, center, radius, (255, 255, 255), -1)
        # cv2.circle(self.image, center, 4, self._color, self.thickness)
        # if label is not None:
        #     text_size = cv2.getTextSize(
        #         label, cv2.FONT_HERSHEY_PLAIN, 1, self.thickness)
        #
        #     center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
        #     pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + \
        #         text_size[0][1]
        #     cv2.rectangle(self.image, pt1, pt2, self._color, self.thickness)
        #     cv2.putText(self.image, label, center, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        # cv2.imwrite("../Traffic_Dataset/%s/video/%d.png" % (video,idx), self.image)

    def line(self, idx, agent_cls, video, x1, y1,w1,h1, x2, y2,w2,h2,x,y,w,h, label=None):
        """Draw a rectangle.

        Parameters
        ----------
        x : float | int
            Top left corner of the rectangle (x-axis).
        y : float | int
            Top let corner of the rectangle (y-axis).
        w : float | int
            Width of the rectangle.
        h : float | int
            Height of the rectangle.
        label : Optional[str]
            A text label that is placed at the top left corner of the
            rectangle.

        """
        pt1 = int(x), int(y)
        pt2 = int(x + w), int(y + h)
        ctr = (int(x) + int(x + w)) / 2, (int(y) + int(y + h)) / 2
        if agent_cls == 1 or agent_cls == 2 or agent_cls == 4:
            pt_lin1 = int((2*x1  +w1) / 2), int(y1)
            pt_lin2 = int((2*x2 +w2) / 2), int(y2)
        elif agent_cls == 3:
            pt_lin1 = int((2*x1  +w1) / 2), int((2*y1  +h1) / 2)
            pt_lin2 = int((2*x2  +w2) / 2), int((2*y2  +h2) / 2)
        else:
            pt_lin1 = int((2*x1 +w1) / 2), int(y1+h1)
            pt_lin2 = int((2*x2 +w2) / 2), int(y2+h2)
        center_txt = int((2*x+w) / 2)-14, int(y)+14

        # cv2.rectangle(self.image, pt1, pt2, self._color, self.thickness)
        # cv2.circle(self.image, center, 4, self._color, self.thickness)
        cv2.line(self.image, (pt_lin1), (pt_lin2), self._color, thickness=3, lineType=cv2.LINE_AA)
        # if label is not None:
            #     text_size = cv2.getTextSize(
            #         label, cv2.FONT_HERSHEY_PLAIN, 1, self.thickness)
            #
            #     center = ctr[0] + 20, ctr[1] + 25 + text_size[0][1]
            #     pt2 = ctr[0] + 40 + text_size[0][0], ctr[1] + 40 + \
            #           text_size[0][1]
            #     # cv2.rectangle(self.image, pt1, pt2, self._color, self.thickness)
            #     cv2.rectangle(self.image, ctr, pt2, self._color, self.thickness)
            # cv2.putText(self.image, label, center_txt, cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 1)
        cv2.imwrite("/scratch1/Research/Aniket_Dataset/%s/video/%d.jpg" % (video, idx), self.image)

    def line_in_cls(self, idx, agent_count, video, x1, y1,w1,h1, x2, y2,w2,h2,x,y,w,h, label=None):
        """Draw a rectangle.

        Parameters
        ----------
        x : float | int
            Top left corner of the rectangle (x-axis).
        y : float | int
            Top let corner of the rectangle (y-axis).
        w : float | int
            Width of the rectangle.
        h : float | int
            Height of the rectangle.
        label : Optional[str]
            A text label that is placed at the top left corner of the
            rectangle.

        """
        if agent_count % 5 == 0:
            pt_lin1 = int((2*x1 + w1) / 2), int(y1 + h1)
            pt_lin2 = int((2*x2 + w2) / 2), int(y2 + h2)
        if agent_count % 5 == 1:
            pt_lin1 = int((2*x1 + w1) / 2), int((4*y1 + h1) / 4)
            pt_lin2 = int((2*x2 + w2) / 2), int((4*y2 + h2) / 4)
        elif agent_count % 5 == 2:
            pt_lin1 = int((2*x1 + w1) / 2), int((2*y1 + h1) / 2)
            pt_lin2 = int((2*x2 + w2) / 2), int((2*y2 + h2) / 2)
        if agent_count % 5 == 3:
            pt_lin1 = int((2*x1 + w1) / 2), int((y1 + 4*h1) / 4)
            pt_lin2 = int((2*x2 + w2) / 2), int((y2 + 4*h2) / 4)
        elif agent_count % 5 == 4:
            pt_lin1 = int((2*x1 + w1) / 2), int(y1)
            pt_lin2 = int((2*x2 + w2) / 2), int(y2)
        # cv2.rectangle(self.image, pt1, pt2, self._color, self.thickness)
        # cv2.circle(self.image, center, 4, self._color, self.thickness)
        # if label is not None:
        #     text_size = cv2.getTextSize(
        #         label, cv2.FONT_HERSHEY_PLAIN, 1, self.thickness)
        #
        #     center = ctr[0] + 20, ctr[1] + 25 + text_size[0][1]
        #     pt2 = ctr[0] + 40 + text_size[0][0], ctr[1] + 40 + \
        #           text_size[0][1]
        #     # cv2.rectangle(self.image, pt1, pt2, self._color, self.thickness)
        #     cv2.rectangle(self.image, ctr, pt2, self._color, self.thickness)
        #     cv2.putText(self.image, label, center, cv2.FONT_HERSHEY_PLAIN, 2, self._color, 2)
        cv2.line(self.image, (pt_lin1), (pt_lin2), self._color, thickness=3, lineType=cv2.LINE_AA)
        cv2.imwrite("../Aniket_Dataset/%s/video/%d.png" % (video, idx), self.image)

    def circle(self, idx,video, x, y, radius, label=None):
        """Draw a circle.

        Parameters
        ----------
        x : float | int
            Center of the circle (x-axis).
        y : float | int
            Center of the circle (y-axis).
        radius : float | int
            Radius of the circle in pixels.
        label : Optional[str]
            A text label that is placed at the center of the circle.

        """
        # image_size = int(radius + self.thickness + 1.5)  # actually half size
        # roi = int(x - image_size), int(y - image_size), \
        #     int(2 * image_size), int(2 * image_size)
        # if not is_in_bounds(self.image, roi):
        #     return

        image = self.image
        # center = image.shape[1] // 2, image.shape[0] // 2
        center = x,y
        # cv2.circle(image, center, int(radius + .5), self._color, self.thickness)
        cv2.circle(self.image, center, radius, self._color, self.thickness)
        # if label is not None:
        #     cv2.putText(
        #         self.image, label, center, cv2.FONT_HERSHEY_PLAIN,
        #         2, self.text_color, 2)
        cv2.imwrite("../Aniket_Dataset/%s/video/%d.png" % (video,idx), self.image)

    def gaussian(self, mean, covariance, label=None):
        """Draw 95% confidence ellipse of a 2-D Gaussian distribution.

        Parameters
        ----------
        mean : array_like
            The mean vector of the Gaussian distribution (ndim=1).
        covariance : array_like
            The 2x2 covariance matrix of the Gaussian distribution.
        label : Optional[str]
            A text label that is placed at the center of the ellipse.

        """
        # chi2inv(0.95, 2) = 5.9915
        vals, vecs = np.linalg.eigh(5.9915 * covariance)
        indices = vals.argsort()[::-1]
        vals, vecs = np.sqrt(vals[indices]), vecs[:, indices]

        center = int(mean[0] + .5), int(mean[1] + .5)
        axes = int(vals[0] + .5), int(vals[1] + .5)
        angle = int(180. * np.arctan2(vecs[1, 0], vecs[0, 0]) / np.pi)
        cv2.ellipse(
            self.image, center, axes, angle, 0, 360, self._color, 2)
        if label is not None:
            cv2.putText(self.image, label, center, cv2.FONT_HERSHEY_PLAIN,
                        2, self.text_color, 2)

    def annotate(self, x, y, text):
        """Draws a text string at a given location.

        Parameters
        ----------
        x : int | float
            Bottom-left corner of the text in the image (x-axis).
        y : int | float
            Bottom-left corner of the text in the image (y-axis).
        text : str
            The text to be drawn.

        """
        cv2.putText(self.image, text, (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN,
                    2, self.text_color, 2)

    def colored_points(self, points, colors=None, skip_index_check=False):
        """Draw a collection of points.

        The point size is fixed to 1.

        Parameters
        ----------
        points : ndarray
            The Nx2 array of image locations, where the first dimension is
            the x-coordinate and the second dimension is the y-coordinate.
        colors : Optional[ndarray]
            The Nx3 array of colors (dtype=np.uint8). If None, the current
            color attribute is used.
        skip_index_check : Optional[bool]
            If True, index range checks are skipped. This is faster, but
            requires all points to lie within the image dimensions.

        """
        if not skip_index_check:
            cond1, cond2 = points[:, 0] >= 0, points[:, 0] < 480
            cond3, cond4 = points[:, 1] >= 0, points[:, 1] < 640
            indices = np.logical_and.reduce((cond1, cond2, cond3, cond4))
            points = points[indices, :]
        if colors is None:
            colors = np.repeat(
                self._color, len(points)).reshape(3, len(points)).T
        indices = (points + .5).astype(np.int)
        self.image[indices[:, 1], indices[:, 0], :] = colors

    def enable_videowriter(self, output_filename, fourcc_string=None,
                           fps=None):
        """ Write images to video file.

        Parameters
        ----------
        output_filename : str
            Output filename.
        fourcc_string : str
            The OpenCV FOURCC code that defines the video codec (check OpenCV
            documentation for more information).
        fps : Optional[float]
            Frames per second. If None, configured according to current
            parameters.

        """
        fourcc = cv2.VideoWriter_fourcc(*fourcc_string)
        if fps is None:
            fps = int(1000. / self._update_ms)
        self._video_writer = cv2.VideoWriter(
            output_filename, fourcc, fps, self._window_shape)

    def disable_videowriter(self):
        """ Disable writing videos.
        """
        self._video_writer = None

    def run(self, update_fun=None):
        """Start the image viewer.

        This method blocks until the user requests to close the window.

        Parameters
        ----------
        update_fun : Optional[Callable[] -> None]
            An optional callable that is invoked at each frame. May be used
            to play an animation/a video sequence.

        """

        if update_fun is not None:
            self._user_fun = update_fun

        self._terminate, is_paused = False, False
        # print("ImageViewer is paused, press space to start.")
        count = 1
        while not self._terminate:
            t0 = time.time()
            if not is_paused:
                self._terminate = not self._user_fun()
                if self._video_writer is not None:
                    self._video_writer.write(
                        cv2.resize(self.image, self._window_shape[:2]))
            t1 = time.time()
            remaining_time = max(1, int(self._update_ms - 1e3*(t1-t0)))
            # cv2.imwrite("../Traffic_Dataset/TRAF16/video/%d.jpg" % (count), cv2.resize(self.image, self._window_shape[:2]))
            # cv2.imshow(self._caption, cv2.resize(self.image, self._window_shape[:2]))

            key = cv2.waitKey(remaining_time)
            if key & 255 == 27:  # ESC
                print("terminating")
                self._terminate = True
            elif key & 255 == 32:  # ' '
                print("toggeling pause: " + str(not is_paused))
                is_paused = not is_paused
            elif key & 255 == 115:  # 's'
                print("stepping")
                self._terminate = not self._user_fun()
                is_paused = True
            count += 1
        # Due to a bug in OpenCV we must call imshow after destroying the
        # window. This will make the window appear again as soon as waitKey
        # is called.
        #
        # see https://github.com/Itseez/opencv/issues/4535
        # self.image[:] = 0
        # cv2.destroyWindow(self._caption)
        # cv2.waitKey(1)

        # cv2.imshow(self._caption, self.image)

    def stop(self):
        """Stop the control loop.

        After calling this method, the viewer will stop execution before the
        next frame and hand over control flow to the user.

        Parameters
        ----------

        """
        self._terminate = True
