import numpy as np

import run_live
import input_reader
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
import torch
import os
from tqdm import tqdm
import cv2
import pandas as pd
import pickle

# OpenPose lightweight to OpenPose
# all_keypoints --> keypoints_list
# pose_entries --> personwise_keypoints
from video_analysis import pose_estimation

fourcc = cv2.VideoWriter_fourcc(*'XVID')
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
video_path = os.path.join(ROOT_DIR, '0_resources', 'beijing2022_fullreplay.mp4')
checkpoint_path = os.path.join(ROOT_DIR, '0_resources', 'checkpoint_iter_370000.pth')
scenes_annotated_path = os.path.join(ROOT_DIR, '0_resources', 'scenes_annotated_01.csv')
scenes_annotated = pd.read_csv(scenes_annotated_path)
synchrony_allscenes = []
normalized_distances_allscenes = []
# shot transition detection: full vid --> scenes

# set up model
net = PoseEstimationWithMobileNet()
checkpoint = torch.load(checkpoint_path, map_location='cpu')
load_state(net, checkpoint)

# call run_live (TODO save synch, add option no output vid)
for index, row in tqdm(scenes_annotated.iterrows(), total=scenes_annotated.shape[0],
                       desc='Progress overall video scenes', leave=False):
    vid = cv2.VideoCapture(video_path)
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vid_writer = cv2.VideoWriter('output%d.avi' % index,  fourcc, fps, (frame_width, frame_height))

    # create video snippet from frames
    for i in tqdm(range(row.frame_start, row.frame_end, 10), leave=False,
                  desc='Progress current video scene (frames)'):
        vid.set(1, i)
        ret, frame = vid.read()
        if ret:
            vid_writer.write(frame)
        else:
            # release vid and vid_writer if ret is false
            vid.release()
            # note when releasing
            print("Released Video Resources")
            # Closes all the frames
            cv2.destroyAllWindows()
            break
    vid_writer.release()
    frame_provider = input_reader.VideoReader('output%d.avi' % index)
    df, synchrony_totalvideo, normalized_distances_totalvideo = run_live.run(net, frame_provider, show_livestream=False,save_outputs=True, synch_style='2pax_90')

    synchrony_allscenes.append(synchrony_totalvideo)
    normalized_distances_allscenes.append(normalized_distances_totalvideo)

    if index == 1:
        break

# create dataframe
col_names = list(("synchrony_" + pose_estimation.keypointsMapping[pose_estimation.POSE_PAIRS[t][0]] + "_to_" +
                      pose_estimation.keypointsMapping[pose_estimation.POSE_PAIRS[t][1]]) for t in
                     range(len(pose_estimation.POSE_PAIRS)))
col_names = col_names[:-2]

final_df = scenes_annotated.copy()
final_df = final_df.iloc[: , 1:] # drop unnamed column
final_df[col_names] =''

final_df['normalized_distance'] = pd.Series(normalized_distances_allscenes)
for i in range(len(synchrony_allscenes)):
    for j in range(17):
        final_df.at[i, col_names[j]] = synchrony_allscenes[i].T[j]


final_df.to_pickle('beijing2022.pkl')