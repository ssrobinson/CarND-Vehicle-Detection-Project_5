import cv2
import numpy as np
from kalman_filters import kalman_predict, kalman_correct
import itertools
import functools
import collections

def bbox_overlap(box1, box2):
    """
    Computes percentage overlap between two bounding boxes
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    area = (x4 - x3 + 1) * (y4 - y3 + 1)

    xx1 = np.maximum(x1, x3)
    yy1 = np.maximum(y1, y3)
    xx2 = np.minimum(x2, x4)
    yy2 = np.minimum(y2, y4)

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    # compute the ratio of overlap
    overlap = (w * h) / area
    return overlap


def centroid(box, as_int=False):
    """
    Computes the centroid of a bounding box
    """
    x1, y1, x2, y2 = box
    if not as_int:
        return ((x1+x2)/2., (y1+y2)/2.)
    else:
        return (int((x1+x2)//2), ((y1+y2)//2))


# Malisiewicz et al.
# http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppression_fast(boxes, overlapThresh):
    """
    Performs non-maximum suppression on a list of bounding boxes
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

class VehicleTracker(object):
    """
    Tracks vehicle candidates using a Kalman filter
    """
    lock_threshold = -50
    def __init__(self, img_size, process_noise = 100.0, measurement_noise = 1000.0, draw_threshold=-35):
        """
        Parameters
        ----------
        img_size : tuple
            The shape of the video frame
        process_noise : float
            Process noise for kalman filter [pixels^2]
        measurement_noise : float
            Measurement covariance in x and y directions [pixels^2]
        """
        self.img_size = img_size
        self.H = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],])
        self.R = np.eye(4)*measurement_noise

        self.F = np.array([[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],])

        self.Q = np.diag([process_noise,process_noise,process_noise,process_noise,
                 process_noise*10, process_noise*10, process_noise*10, process_noise*10,
                 process_noise*100, process_noise*100, process_noise*100, process_noise*100])

        self.P0 = np.eye(12)*10e3  # Initial covariance of any new candidate
        self.predict = functools.partial(kalman_predict, F = self.F, Q = self.Q)
        self.correct = functools.partial(kalman_correct, H = self.H, R = self.R)

        self.draw_threshold = draw_threshold
        self.candidates = []

    def create_candidate(self, meas):
        """
        Creates a new tracking candidate from a measurement (bounding box)
        Parameters

        meas - tuple
            tuple of 4 numbers denoting the top-left and bottom-right corners of the box
        """
        # Position
        x = list(meas)
        # Velocity and Acceleration
        mag = 0.05
        x.extend([-mag*2 if x[0] > self.img_size[1]/2 else mag*5,
                  -mag,
                  -mag*2 if x[0] > self.img_size[1]/2 else mag*5,
                  -mag,
                  0.00, 0.00, 0.00, 0.00])

        return {'x': np.array(x), 'P': np.copy(self.P0), 'age': 0}

    def track(self, measurements):
        """
        Uses measurements from OpenCV detect vehicles in the video stream
        Parameters

        measurements : list
            List of bounding boxes detected by the classifier (and heatmap)
        """
        measurements = non_max_suppression_fast(np.array(measurements), overlapThresh=0.01)
        # measurements = np.array(measurements)
        meas_centroid = np.array([centroid(meas) for meas in measurements])
        if len(self.candidates) == 0:
            self.candidates = [self.create_candidate(meas) for meas in measurements]
        else:
            # Assign measurements to candidates based on overlap
            cand_measurements = collections.defaultdict(list)
            for j, (meas, cent) in enumerate(zip(measurements, meas_centroid)):
                # Overlap with each candidate
                overlap = [bbox_overlap(cand['x'][:4], meas) for cand in self.candidates]
                # Distance to each candidate
                distance = [np.linalg.norm(centroid(cand['x'][:4]) - cent) for cand in self.candidates]

                max_cand_idx = np.argmax(overlap)
                min_dist_idx = np.argmin(distance)

                if overlap[max_cand_idx] < 0.01 and distance[max_cand_idx] > 150:
                    self.candidates.append(self.create_candidate(meas))
                else:
                    # Assign measurement to candidate
                    if overlap[max_cand_idx] >= 0.01:
                        cand_measurements[max_cand_idx].append(meas)
                    else:
                        cand_measurements[min_dist_idx].append(meas)
            for i, cand in enumerate(self.candidates):
                meas_list = cand_measurements[i]
                x1, P1 = self.predict(x = cand['x'], P = cand['P'])
                if meas_list:
                    meas = sum(meas_list)/len(meas_list)
                    x1, P1 = self.correct(z = meas, x = x1, P = P1)



                # Increment age if no measurements matched this candidate
                if cand['age'] > self.lock_threshold:
                    if not meas_list:
                        self.candidates[i]['age'] = cand['age'] + 1
                    else:
                        self.candidates[i]['age'] = cand['age'] - 1
                self.candidates[i]['x'] = x1
                self.candidates[i]['P'] = P1


    def draw_bboxes(self, image):

        for c in self.candidates:
            bbox = c['x'][:4].astype(np.int32)
            cov = np.sqrt(np.trace(c['P'][:4]))
            if c['age'] > self.draw_threshold and cov > 70.0:
                continue
            else:
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 3)

        return image
