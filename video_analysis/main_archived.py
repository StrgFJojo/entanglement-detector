import sys

import data_optimization
import pose_estimation
import distance_measure
import height_measure
import synchrony_measure
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
video_path = '/Users/josephinevandelden/PycharmProjects/entanglement/video_prep/Figure Skating - Pairs Short Program  ' \
             'Full Replay  #Beijing2022.mp4'
scenes_annotated_path = os.path.join(ROOT_DIR, '0_resources', 'scenes_annotated_01.csv')

protoFile = "/Users/josephinevandelden/Documents/GitHub/learnopencv/OpenPose-Multi-Person/pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "/Users/josephinevandelden/Downloads/pose_iter_440000.caffemodel"


def run_analysis(vid, frame_start, frame_end):
    distances_totalvideo = -1 * np.ones(2)
    heights_totalvideo = -1 * np.ones(3)
    synchrony_totalvideo = -1 * np.ones(18)
    normalized_distances_totalvideo = np.ones_like(distances_totalvideo)

    for i in range(frame_start, frame_end,30):
        print("Processing frame", i)
        vid.set(1, i)
        #hasFrame, frame1 = vid.read()
        # set counter to initial frame so that loop can skip frames
        count = 1

        # needed?
        # Create output file
        #vid_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
        #                             (frame1.shape[1], frame1.shape[0]))

        # reads network model (caffe framework)
        # params: prototxt -> text description of architecture, caffeModel -> learned network
        net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

        # not using cuda or tensorflow here
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        # print("Start using CPU device")

        #while (True):

        # Capture frame by frame
        # tuple (bool, frame)
        # use boolean to check if frames are there or not
        ret, frame = vid.read()
        if not ret:
            # release vid if ret is false
            vid.release()
            # also release output video vid_writer
           # vid_writer.release()
            # note when releasing
            print("Released Video Resource")
            # Closes all the frames
            cv2.destroyAllWindows()
            break

        # process every xth frame only
        # TODO find a cleaner way to do this
        count += 1
        if count % 30 != 0:
            continue


        print("Processing frame", count)
        #plt.figure(figsize=[15, 15])
        #plt.imshow(frame[:, :, [2, 1, 0]])
        #plt.show()
        frameClone, keypoints_list, detected_keypoints, personwiseKeypoints = pose_estimation.get_poses(frame, net)
        if len(keypoints_list) == 0:
            print("Frame %d skipped. No keypoints detected." % count)
            continue

        vid_writer.write(frameClone)

        # find 2 most prominent persons in picture
        scores = np.zeros(len(personwiseKeypoints))

        for j in range(len(personwiseKeypoints)):
            scores[j] = personwiseKeypoints[j][-1]

        indices_of_two_highest_scores = (-scores).argsort()[:2]

        # Proximity

        distance_midhip = distance_measure.get_distance(personwiseKeypoints, indices_of_two_highest_scores,
                                                        keypoints_list)
        distances_totalvideo = np.vstack((distances_totalvideo, np.hstack((count, distance_midhip))))

        # Height
        heights = height_measure.get_height(indices_of_two_highest_scores, personwiseKeypoints, keypoints_list)
        heights_totalvideo = np.vstack((heights_totalvideo, np.hstack((count, heights))))

        # Synchrony
        synch_degree = synchrony_measure.get_synchrony(indices_of_two_highest_scores, personwiseKeypoints,
                                                       pose_estimation.POSE_PAIRS, keypoints_list, count)
        synchrony_totalvideo = np.vstack((synchrony_totalvideo, np.hstack((count, synch_degree))))

    # remove leading -1 row
    synchrony_totalvideo = synchrony_totalvideo[1:][:]
    heights_totalvideo = heights_totalvideo[1:][:]
    distances_totalvideo = distances_totalvideo[1:][:]
    print(distances_totalvideo)
    for k in range(distances_totalvideo.shape[0]):
        normalized_distances_totalvideo[k][0] = distances_totalvideo[k][0]  # frame num
        if -1 in heights_totalvideo[k][:]:
            normalized_distances_totalvideo[k][1] = -1
        else:
            normalized_distances_totalvideo[k][1] = distances_totalvideo[k][1] / (
                    heights_totalvideo[k][1] + heights_totalvideo[k][2])

    return synchrony_totalvideo, normalized_distances_totalvideo


# Grab part of the input video
# run pose estimation and synch / proximity calculation
# attach time series to dataframe

scenes_annotated = pd.read_csv(scenes_annotated_path)
scenes_annotated["synchrony_timeseries"] = " "
scenes_annotated["normalized_distance_timeseries"] = " "

for index, row in tqdm(scenes_annotated.iterrows(), total=scenes_annotated.shape[0], desc='Processing video scenes'):
    vid = cv2.VideoCapture(video_path)
    synchrony_totalvideo, normalized_distances_totalvideo = run_analysis(vid, row.frame_start, row.frame_end)
    row.synchrony_timeseries = synchrony_totalvideo
    row.normalized_distance_timeseries = normalized_distances_totalvideo

print("xx")

