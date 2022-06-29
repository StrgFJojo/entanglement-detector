import time
import warnings
import visualization
import pose_estimation
import distance_measure
import height_measure
import synchrony_measure
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import os

warnings.filterwarnings('ignore')

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
video_path = os.path.join(ROOT_DIR, '0_resources', 'beijing2022_fullreplay.mp4')
scenes_annotated_path = os.path.join(ROOT_DIR, '0_resources', 'scenes_annotated_01.csv')
protoFile = os.path.join(ROOT_DIR, '0_resources', 'pose_deploy_linevec.prototxt')
weightsFile = os.path.join(ROOT_DIR, '0_resources', 'pose_iter_440000.caffemodel')

scenes_annotated = pd.read_csv(scenes_annotated_path)
scenes_annotated["synchrony_timeseries"] = " "
scenes_annotated["normalized_distance_timeseries"] = " "

for index, row in tqdm(scenes_annotated.iterrows(), total=scenes_annotated.shape[0],
                       desc='Progress overall video scenes', leave=False):
    #if index == 3:
    #    break
    vid = cv2.VideoCapture(video_path)
    ret0, frame0 = vid.read()

    distances_totalvideo = -1 * np.ones(2)
    heights_totalvideo = -1 * np.ones(3)
    synchrony_totalvideo = -1 * np.ones(18)

    vid_writer = cv2.VideoWriter('output%d.avi' % index, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                (frame0.shape[1], frame0.shape[0]))

    for i in tqdm(range(row.frame_start, row.frame_end, 30), leave=False,
                  desc='Progress current video scene (frames)', ):
        # print("Processing frame", i)
        vid.set(1, i)

        # reads network model (caffe framework)
        # params: prototxt -> text description of architecture, caffeModel -> learned network
        net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

        # Capture frame
        # tuple (bool, frame)
        # use boolean to check if frames are there or not
        ret, frame = vid.read()

        if not ret:
            # release vid if ret is false
            vid.release()
            # note when releasing
            print("Released Video Resource")
            # Closes all the frames
            cv2.destroyAllWindows()
            break

        frameClone, keypoints_list, detected_keypoints, personwiseKeypoints = pose_estimation.get_poses(frame, net,
                                                                                                        index, i)

        if len(personwiseKeypoints) < 2:
            # print("Frame %d skipped. Keypoints for %d persons detected." % (i, len(personwiseKeypoints)))
            continue

        # find 2 most prominent persons in picture
        scores = np.zeros(len(personwiseKeypoints))

        for j in range(len(personwiseKeypoints)):
            scores[j] = personwiseKeypoints[j][-1]

        indices_of_two_highest_scores = (-scores).argsort()[:2]

        # Proximity

        distance_midhip = distance_measure.get_distance(personwiseKeypoints, indices_of_two_highest_scores,
                                                        keypoints_list)
        distances_totalvideo = np.vstack((distances_totalvideo, np.hstack((i, distance_midhip))))

        # Height
        heights = height_measure.get_height(indices_of_two_highest_scores, personwiseKeypoints, keypoints_list)
        heights_totalvideo = np.vstack((heights_totalvideo, np.hstack((i, heights))))

        # Synchrony
        synch_degree = synchrony_measure.get_synchrony(indices_of_two_highest_scores, personwiseKeypoints,
                                                       pose_estimation.POSE_PAIRS, keypoints_list)
        synchrony_totalvideo = np.vstack((synchrony_totalvideo, np.hstack((i, synch_degree))))

        #frameClone=visualization.full_skeleton_w_synchdegree(personwiseKeypoints, pose_estimation.POSE_PAIRS, keypoints_list, synch_degree, frameClone)
        frameClone = visualization.partly_overlay(personwiseKeypoints, pose_estimation.POSE_PAIRS,
                                                               keypoints_list, synch_degree, frameClone)
        vid_writer.write(frameClone)

    # release output video vid_writer
    vid_writer.release()
    # remove leading -1 row
    synchrony_totalvideo = synchrony_totalvideo[1:][:]
    heights_totalvideo = heights_totalvideo[1:][:]
    distances_totalvideo = distances_totalvideo[1:][:]
    normalized_distances_totalvideo = np.ones_like(distances_totalvideo)

    for k in range(distances_totalvideo.shape[0]):
        normalized_distances_totalvideo[k][0] = distances_totalvideo[k][0]  # frame num
        if -1 in heights_totalvideo[k][:]:
            normalized_distances_totalvideo[k][1] = -1
        else:
            normalized_distances_totalvideo[k][1] = distances_totalvideo[k][1] / (
                    heights_totalvideo[k][1] + heights_totalvideo[k][2])

    scenes_annotated.at[index, 'synchrony_timeseries'] = synchrony_totalvideo
    scenes_annotated.at[index, 'normalized_distance_timeseries'] = normalized_distances_totalvideo

scenes_annotated.to_csv(os.path.join(ROOT_DIR, '0_resources', 'scenes_annotated_02.csv'))
