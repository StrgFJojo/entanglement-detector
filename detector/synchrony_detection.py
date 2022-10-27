import math
from typing import List

import numpy as np

from detector.pose_estimation import Pose, PoseEstimator

column_names = [
    f"synch_{Pose.kpt_names[keypoint_pair[0]]}"
    f"_to_{Pose.kpt_names[keypoint_pair[1]]}"
    for keypoint_pair in PoseEstimator.skeleton_keypoint_pairs
]


class SynchronyDetector:
    """
    Takes pose list and calculates synchronization of
    the individual body parts according to the specified synch metric.
    """

    def __init__(self, synch_metric: str):
        self.synch_metric = synch_metric
        self.synchrony = []

    def calculate_synchrony(self, poses):
        synch_scores = np.nan * np.empty(17)
        # return empty list if there is less than two bodies
        if poses is None or len(poses) < 2:
            synch_dict = dict(zip(column_names, synch_scores))
            self.synchrony.append(synch_dict)
            return synch_dict

        # synchronization calculation if there is more than two bodies
        for bodypart_idx, keypoint_pair in enumerate(
            PoseEstimator.skeleton_keypoint_pairs
        ):
            angs = 0
            count = 0
            for pose_idx1, pose1 in enumerate(poses):
                for pose_idx2, pose2 in enumerate(poses):
                    if pose_idx2 > pose_idx1:
                        unit_vectors = []
                        for idx, pose in enumerate([pose1, pose2]):
                            # get same- or opposite-side keypoints of body part
                            kpt1, kpt2 = self.get_keypoints_of_bodypart(
                                pose, idx, keypoint_pair, bodypart_idx
                            )
                            # transform keypoints to unit vectors
                            unit_vectors.append(
                                self.keypoints_to_unit_vector(kpt1, kpt2)
                            )
                        unit_vectors = [
                            x for x in unit_vectors if np.nan not in x
                        ]
                        if len(unit_vectors) >= 2:
                            angs += self.get_angle(
                                unit_vectors[0], unit_vectors[1]
                            )
                            count += 1
            if count == 0:
                synch_score = np.nan
            else:
                avg_ang = angs / count
                synch_score = self.angle_to_synch_score(avg_ang)
            synch_scores[bodypart_idx] = synch_score
        synch_dict = dict(zip(column_names, synch_scores))
        self.synchrony.append(synch_dict)
        return synch_dict

    @staticmethod
    def keypoints_to_unit_vector(keypoint1, keypoint2):
        if -1 in keypoint1 or -1 in keypoint2:
            unit_vector = [np.nan, np.nan]
        else:
            x1 = keypoint1[0]
            y1 = keypoint1[1]
            x2 = keypoint2[0]
            y2 = keypoint2[1]
            distance = [x2 - x1, y2 - y1]
            norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2)
            unit_vector = [
                distance[0] / norm,
                distance[1] / norm,
            ]
        return unit_vector

    @staticmethod
    def get_angle(vector1: List[float], vector2: List[float]) -> float:
        dot_product = np.dot(vector1, vector2)
        rad = np.arccos(dot_product)  # in radians
        deg = np.rad2deg(rad)  # in degree
        return deg

    def angle_to_synch_score(self, deg: float) -> float:
        if self.synch_metric in ["pss", "pos"]:
            if deg <= 90:
                synch_score = 1 + ((0 - 1) / (90 - 0)) * (deg - 0)
            else:
                synch_score = 0 + ((1 - 0) / (180 - 90)) * (deg - 90)
        else:  # self.synch_metric in ['lss', 'los']
            synch_score = 1 - deg / 180
        return synch_score

    def vector_diff_to_synch_score(self, unit_vectors):
        unit_vectors = [x for x in unit_vectors if np.nan not in x]
        if len(unit_vectors) < 2:
            synch_score = np.nan
        else:
            angs = 0
            count = 0
            for idx1, vec1 in enumerate(unit_vectors):
                for idx2, vec2 in enumerate(unit_vectors):
                    if idx2 > idx1:
                        angs += self.get_angle(vec1, vec2)
                        count += 1
            avg_ang = angs / count
            synch_score = self.angle_to_synch_score(avg_ang)
        return synch_score

    def get_keypoints_of_bodypart(
        self, pose, person_idx, keypoint_pair, bodypart_idx
    ):
        if self.synch_metric in ["pos", "los"] and person_idx == 1:
            keypoint1 = pose.keypoints[
                PoseEstimator.skeleton_keypoint_pairs_mirrored[bodypart_idx][0]
            ]
            keypoint2 = pose.keypoints[
                PoseEstimator.skeleton_keypoint_pairs_mirrored[bodypart_idx][1]
            ]
        else:
            keypoint1 = pose.keypoints[keypoint_pair[0]]
            keypoint2 = pose.keypoints[keypoint_pair[1]]
        return keypoint1, keypoint2

    def get_relevant_poses(self, poses, group_size):
        if len(poses) < 2:
            relevant_poses = None
        else:
            conf_vals = []
            for pose in poses:
                conf_vals.append(pose.confidence)
            if group_size == "all":
                pax = len(poses)
            else:
                pax = int(group_size)
            relevant_poses_idx = (-np.array(conf_vals)).argsort()[:pax]
            relevant_poses = [
                pose
                for idx, pose in enumerate(poses)
                if idx in relevant_poses_idx
            ]
        return relevant_poses

    @staticmethod
    def get_mirror_key(dict_key):
        org_idx = column_names.index(dict_key)
        mirror_idx = PoseEstimator.idx_mirror_pose_pair[org_idx]
        return column_names[mirror_idx]
