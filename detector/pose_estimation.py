import math

import cv2
import numpy as np
import torch

from modules.keypoints import (
    BODY_PARTS_KPT_IDS,
    BODY_PARTS_PAF_IDS,
    extract_keypoints,
    group_keypoints,
)
from modules.one_euro_filter import OneEuroFilter


class PoseEstimator:
    """
    Takes input image and estimates poses.
    """

    skeleton_keypoint_pairs = [
        [1, 2],
        [1, 5],
        [2, 3],
        [3, 4],
        [5, 6],
        [6, 7],
        [1, 8],
        [8, 9],
        [9, 10],
        [1, 11],
        [11, 12],
        [12, 13],
        [1, 0],
        [0, 14],
        [14, 16],
        [0, 15],
        [15, 17],
    ]
    idx_mirror_pose_pair = [
        1,
        0,
        4,
        5,
        2,
        3,
        9,
        10,
        11,
        6,
        7,
        8,
        12,
        15,
        16,
        13,
        14,
    ]

    skeleton_keypoint_pairs_mirrored = [
        [1, 5],
        [1, 2],
        [5, 6],
        [6, 7],
        [2, 3],
        [3, 4],
        [1, 11],
        [11, 12],
        [12, 13],
        [1, 8],
        [8, 9],
        [9, 10],
        [1, 0],
        [0, 15],
        [15, 17],
        [0, 14],
        [14, 16],
    ]

    def __init__(self, net, height_size, stride=8, upsample_ratio=4, cpu=True):
        self.net = net
        self.height_size = height_size
        self.stride = stride
        self.upsample_ratio = upsample_ratio
        self.cpu = cpu
        self.pad_value = (0, 0, 0)
        self.img_mean = np.array([128, 128, 128], np.float32)
        self.img_scale = np.float32(1 / 256)

    def img_to_poses(self, img):
        current_poses = []
        heatmaps, pafs, scale, pad = self.infer_fast(img)
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(Pose.num_kpts):  # 19th for bg
            total_keypoints_num += extract_keypoints(
                heatmaps[:, :, kpt_idx],
                all_keypoints_by_type,
                total_keypoints_num,
            )

        pose_entries, all_keypoints = group_keypoints(
            all_keypoints_by_type, pafs
        )

        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (
                all_keypoints[kpt_id, 0] * self.stride / self.upsample_ratio
                - pad[1]
            ) / scale
            all_keypoints[kpt_id, 1] = (
                all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio
                - pad[0]
            ) / scale
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((Pose.num_kpts, 2), dtype=np.int32) * -1
            for kpt_id in range(Pose.num_kpts):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 0]
                    )
                    pose_keypoints[kpt_id, 1] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 1]
                    )
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)
        return current_poses

    def infer_fast(self, img):
        height, width, _ = img.shape
        scale = self.height_size / height

        scaled_img = cv2.resize(
            img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        scaled_img = self.normalize(img=scaled_img)
        min_dims = [
            self.height_size,
            max(scaled_img.shape[1], self.height_size),
        ]
        padded_img, pad = self.pad_width(img=scaled_img, min_dims=min_dims)

        tensor_img = (
            torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        )
        if not self.cpu:
            tensor_img = tensor_img.cpu()

        stages_output = self.net(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(
            stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0)
        )
        heatmaps = cv2.resize(
            heatmaps,
            (0, 0),
            fx=self.upsample_ratio,
            fy=self.upsample_ratio,
            interpolation=cv2.INTER_CUBIC,
        )

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(
            pafs,
            (0, 0),
            fx=self.upsample_ratio,
            fy=self.upsample_ratio,
            interpolation=cv2.INTER_CUBIC,
        )

        return heatmaps, pafs, scale, pad

    def normalize(self, img):
        img = np.array(img, dtype=np.float32)
        img = (img - self.img_mean) * self.img_scale
        return img

    def pad_width(self, img, min_dims):
        h, w, _ = img.shape
        h = min(min_dims[0], h)
        min_dims[0] = math.ceil(min_dims[0] / float(self.stride)) * self.stride
        min_dims[1] = max(min_dims[1], w)
        min_dims[1] = math.ceil(min_dims[1] / float(self.stride)) * self.stride
        pad = []
        pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
        pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
        pad.append(int(min_dims[0] - h - pad[0]))
        pad.append(int(min_dims[1] - w - pad[1]))
        padded_img = cv2.copyMakeBorder(
            img,
            pad[0],
            pad[2],
            pad[1],
            pad[3],
            cv2.BORDER_CONSTANT,
            value=self.pad_value,
        )
        return padded_img, pad


class Pose:
    num_kpts = 18
    kpt_names = [
        "nose",
        "neck",
        "r_sho",
        "r_elb",
        "r_wri",
        "l_sho",
        "l_elb",
        "l_wri",
        "r_hip",
        "r_knee",
        "r_ank",
        "l_hip",
        "l_knee",
        "l_ank",
        "r_eye",
        "l_eye",
        "r_ear",
        "l_ear",
    ]

    sigmas = (
        np.array(
            [
                0.26,
                0.79,
                0.79,
                0.72,
                0.62,
                0.79,
                0.72,
                0.62,
                1.07,
                0.87,
                0.89,
                1.07,
                0.87,
                0.89,
                0.25,
                0.25,
                0.35,
                0.35,
            ],
            dtype=np.float32,
        )
        / 10.0
    )
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [0, 224, 255]

    def __init__(self, keypoints, confidence):
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        self.bbox = Pose.get_bbox(self.keypoints)
        self.id = None
        self.filters = [
            [OneEuroFilter(), OneEuroFilter()] for _ in range(Pose.num_kpts)
        ]

    @staticmethod
    def get_bbox(keypoints):
        found_keypoints = np.zeros(
            (np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32
        )
        found_kpt_id = 0
        for kpt_id in range(Pose.num_kpts):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        bbox = cv2.boundingRect(found_keypoints)
        return bbox

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = Pose.last_id + 1
            Pose.last_id += 1

    def draw(self, img, colors):
        assert self.keypoints.shape == (Pose.num_kpts, 2)

        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            color = colors[part_id]
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                cv2.circle(img, (int(x_a), int(y_a)), 3, color, -1)
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]
            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                cv2.circle(img, (int(x_b), int(y_b)), 3, color, -1)
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                cv2.line(
                    img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), color, 2
                )


def get_similarity(a, b, threshold=0.5):
    num_similar_kpt = 0
    for kpt_id in range(Pose.num_kpts):
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
            distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            similarity = np.exp(
                -distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id])
            )
            if similarity > threshold:
                num_similar_kpt += 1
    return num_similar_kpt


def track_poses(previous_poses, current_poses, threshold=3, smooth=False):
    """Propagate poses ids from previous frame results. Id is propagated,
    if there are at least `threshold` similar keypoints between pose from
    previous frame and current.
    If correspondence between pose on previous and current frame was
    established, pose keypoints are smoothed.

    :param previous_poses: poses from previous frame with ids
    :param current_poses: poses from current frame to assign ids
    :param threshold: minimal number of similar keypoints between poses
    :param smooth: smooth pose keypoints between frames
    :return: None
    """
    current_poses = sorted(
        current_poses, key=lambda pose: pose.confidence, reverse=True
    )  # match confident poses first
    mask = np.ones(len(previous_poses), dtype=np.int32)

    for current_pose in current_poses:
        best_matched_id = None
        best_matched_pose_id = None
        best_matched_iou = 0
        for id, previous_pose in enumerate(previous_poses):
            if not mask[id]:
                continue
            iou = get_similarity(current_pose, previous_pose)
            if iou > best_matched_iou:
                best_matched_iou = iou
                best_matched_pose_id = previous_pose.id
                best_matched_id = id
        if best_matched_iou >= threshold:
            mask[best_matched_id] = 0
        else:  # pose not similar to any previous
            best_matched_pose_id = None
        current_pose.update_id(best_matched_pose_id)

        if smooth:
            for kpt_id in range(Pose.num_kpts):
                if current_pose.keypoints[kpt_id, 0] == -1:
                    continue
                # reuse filter if previous pose has valid filter
                if (
                    best_matched_pose_id is not None
                    and previous_poses[best_matched_id].keypoints[kpt_id, 0]
                    != -1
                ):
                    current_pose.filters[kpt_id] = previous_poses[
                        best_matched_id
                    ].filters[kpt_id]
                current_pose.keypoints[kpt_id, 0] = current_pose.filters[
                    kpt_id
                ][0](current_pose.keypoints[kpt_id, 0])
                current_pose.keypoints[kpt_id, 1] = current_pose.filters[
                    kpt_id
                ][1](current_pose.keypoints[kpt_id, 1])
            current_pose.bbox = Pose.get_bbox(current_pose.keypoints)
