import os
import argparse
import warnings
import cv2
import numpy as np
import torch
import math
import pandas as pd
from datetime import datetime
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose_estimation_lightweight import Pose, track_poses
from detector import distance_measure, synchrony_measure,height_measure,visualization,input_reader
from modules import pose_estimation_openpose

warnings.filterwarnings('ignore')


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1 / 256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cpu()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run(net, image_provider, height_size=256, cpu=True, track=1, smooth=1, show_livestream=True, save_livestream=False,
        save_outputs=False, synch_style='2pax_90', full_stats_mode = 0):

    if full_stats_mode:
        show_livestream = False
        save_outputs = True
        synch_style_list = ['2pax_90', '2pax_180', '2pax_90_mirrored', '2pax_180_mirrored', '2']
        print("Running in full stats mode. Livestream switched off per default. Outputs saved to CSV.")

    if save_livestream:
        iterable = iter(image_provider)
        ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
        output_video_path = os.path.join(ROOT_DIR, 'olympics/per_scene_entanglement_visualized', 'output.avi')
        frame_width = int(iterable.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(iterable.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(iterable.cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    quit_flag = False
    net = net.eval()
    if not cpu:
        net = net.cpu()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1
    idx = 0
    normalized_distances_totalvideo = -1 * np.ones(1)
    synchrony_totalvideo = -1 * np.ones(17)
    distance_midhip = -1

    flag = True

    for img in image_provider:
        synch_degree = -1 * np.ones(17)
        person_indices = [-1,-1]
        if quit_flag:
            break
        orig_img = img.copy()
        start = datetime.now().time()  # time object
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)
        end = datetime.now().time()
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)

        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if len(pose_entries) > 1:

            # find most prominent persons in picture
            scores = np.zeros(len(pose_entries))
            for j in range(len(pose_entries)):
                scores[j] = pose_entries[j][18]
            if synch_style in ['2pax_90', '2pax_180', '2pax_90_mirrored', '2pax_180_mirrored']:
                pax = 2
            elif synch_style == 'allpax':
                pax = len(pose_entries)
            else:
                try:
                    pax = int(synch_style)
                except ValueError:
                    print('%s is not a valid input for argument synch_style' % synch_style)
                    exit(1)
            person_indices = (-scores).argsort()[:pax]

            synch_degree = synchrony_measure.get_synchrony_flexpax(pose_entries, person_indices,
                                                                   pose_estimation_openpose.POSE_PAIRS, all_keypoints,
                                                                   synch_style)
            distance_midhip = distance_measure.get_distance_multipax(pose_entries, all_keypoints, person_indices)
            heights = height_measure.get_height_multipax(pose_entries, all_keypoints, person_indices)

            if -1 in heights:
                normalized_distance = -1
            else:
                normalized_distance = distance_midhip / (
                        sum(heights))

            if save_outputs:
                synchrony_totalvideo = np.vstack((synchrony_totalvideo, synch_degree))
                normalized_distances_totalvideo = np.append(normalized_distances_totalvideo, normalized_distance)
        else:
            if save_outputs:  # -1
                synchrony_totalvideo = np.vstack((synchrony_totalvideo, -1 * np.ones(17)))
                normalized_distances_totalvideo = np.append(normalized_distances_totalvideo, -1)
        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        if show_livestream or save_livestream:
            #for idx, pose in enumerate(current_poses):
            #    pose.draw(img, synch_degree,person_indices, synch_style, idx)

            img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)

            for pose in current_poses:
                cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                              (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
                if track:
                    cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

            img = visualization.partly_overlay_final(pose_entries, person_indices,
                                                     pose_estimation_openpose.POSE_PAIRS,
                                                     all_keypoints, synch_degree, distance_midhip, img,
                                                     synch_style)
            if show_livestream:
                cv2.imshow('Synch Detector', img)
                key = cv2.waitKey(delay)
                if key == 27:  # esc
                    cv2.destroyAllWindows()
                    for i in range(1, 5):
                        cv2.waitKey(1)
                        quit_flag = True
                        break
                elif key == 112:  # 'p'
                    if delay == 1:
                        delay = 0
                    else:
                        delay = 1

            if save_livestream:
                writer.write(img)

    if show_livestream:
        cv2.destroyAllWindows()
    if save_livestream:
        writer.release()
    idx += 1
    if save_outputs:
        synchrony_totalvideo = synchrony_totalvideo[1:][:]
        normalized_distances_totalvideo = normalized_distances_totalvideo[1:][:]

        col_names = list(("synchrony_" + pose_estimation_openpose.keypointsMapping[pose_estimation_openpose.POSE_PAIRS[t][0]] + "_to_" +
                          pose_estimation_openpose.keypointsMapping[pose_estimation_openpose.POSE_PAIRS[t][1]]) for t in
                         range(len(pose_estimation_openpose.POSE_PAIRS)))
        col_names = col_names[:-2]
        df = pd.DataFrame(columns=col_names)
        df['normalized_distance'] = pd.Series(normalized_distances_totalvideo)
        for i in range(len(synchrony_totalvideo)):
            for j in range(17):
                df.at[i, col_names[j]] = synchrony_totalvideo[i].T[j]
        return df, synchrony_totalvideo, normalized_distances_totalvideo
    else:
        return


if __name__ == '__main__':
    ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
    checkpoint_path = os.path.join(ROOT_DIR, 'models', 'checkpoint_iter_370000.pth')

    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    parser.add_argument('--show-livestream', default=True)
    parser.add_argument('--save-livestream', default=False)
    parser.add_argument('--save-outputs', default=False)
    parser.add_argument('--synch-style', type=str, default='2pax_90', help='type of synchrony metric')
    parser.add_argument('--full-stats-mode', type=int, default=0, help='get all synch styles')
    parser.add_argument('--checkpoint-path', type=str, default=checkpoint_path)
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')


    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    frame_provider = input_reader.ImageReader(args.images)
    if args.video != '':
        frame_provider = input_reader.VideoReader(args.video)
    else:
        args.track = 0

    if args.save_outputs:
        df, synchrony_totalvideo, normalized_distances_totalvideo = run(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth, args.show_livestream, args.save_outputs, args.synch_style, args.full_stats_mode)
        df.to_csv('output_table.csv', index=True, index_label='Frame_id')
    else:
        run(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth, args.show_livestream,
            args.save_outputs, args.synch_style, args.full_stats_mode)

    exit()
