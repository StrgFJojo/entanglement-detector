import math
from statistics import mean
from typing import List

import numpy as np

from detector.pose_estimation import Pose, PoseEstimator


class SynchronyDetector:
    """
    Takes pose list and calculates synchronization of
    the individual body parts according to the specified synch metric.
    """

    synch_styles_mirrored = ["2pax_90_mirrored", "2pax_180_mirrored"]
    synch_styles_2persons = [
        "2pax_90",
        "2pax_180",
        "2pax_90_mirrored",
        "2pax_180_mirrored",
    ]
    synch_styles_90 = ["2pax_90", "2pax_90_mirrored"]
    synch_styles_excl_int_params = [
        "2pax_90",
        "2pax_180",
        "2pax_90_mirrored",
        "2pax_180_mirrored",
        "allpax",
    ]

    def __init__(self, synch_style: str):
        self.synch_style = synch_style
        self.synchrony = []
        self.column_names = [
            f"synchrony_{Pose.kpt_names[keypoint_pair[0]]}"
            f"_to_{Pose.kpt_names[keypoint_pair[1]]}"
            for keypoint_pair in PoseEstimator.skeleton_keypoint_pairs
        ]

    def calculate_synchrony(self, poses):
        synch_scores = -1 * np.ones(17)
        if poses is None or len(poses) < 2:
            self.synchrony.append(dict(zip(self.column_names, synch_scores)))
        else:
            for bodypart_idx, keypoint_pair in enumerate(
                PoseEstimator.skeleton_keypoint_pairs
            ):

                unit_vectors = -1 * np.ones((len(poses), 2))

                for person_idx, pose in enumerate(poses):
                    keypoint1, keypoint2 = self.get_keypoints_of_bodypart(
                        pose, person_idx, keypoint_pair, bodypart_idx
                    )
                    unit_vectors[person_idx] = self.keypoints_to_unit_vector(
                        keypoint1, keypoint2
                    )

                if self.synch_style in self.synch_styles_2persons:
                    if (-1 in unit_vectors[0]) or (-1 in unit_vectors[1]):
                        synch_scores[bodypart_idx] = -1
                    else:
                        deg = self.get_angle(unit_vectors[0], unit_vectors[1])
                        synch_score = self.angle_to_synch_score(deg)
                        synch_scores[bodypart_idx] = synch_score
                else:
                    synch_scores[
                        bodypart_idx
                    ] = self.vector_diff_to_synch_score(unit_vectors)

            synch_scores = self.normalize_synch_scores(synch_scores)
            synch_dict = dict(zip(self.column_names, synch_scores))
            self.synchrony.append(synch_dict)
            return synch_dict

    @staticmethod
    def keypoints_to_unit_vector(keypoint1, keypoint2):
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
        if self.synch_style in self.synch_styles_90:
            if deg <= 90:
                synch_score = 1 + ((0 - 1) / (90 - 0)) * (deg - 0)
            else:
                synch_score = 0 + ((1 - 0) / (180 - 90)) * (deg - 90)
        else:  # synch_style == '2pax_180'
            synch_score = 1 - deg / 180
        return synch_score

    def vector_diff_to_synch_score(self, unit_vectors):
        unit_vectors = [x for x in unit_vectors if -1 not in x]
        vector_sum = [0, 0]
        for vec in unit_vectors:
            vector_sum += vec
        if len(unit_vectors) > 1:
            avg_vec = [x / len(unit_vectors) for x in vector_sum]
            dif_vectors_sum = 0
            for j in range(len(unit_vectors)):
                dif_vectors_sum += abs(unit_vectors[j] - avg_vec)
            synch_score = mean(dif_vectors_sum)
        else:
            synch_score = -1
        return synch_score

    def normalize_synch_scores(self, synch_scores):
        if self.synch_style not in self.synch_styles_2persons:
            synch_scores = [
                (float(x) / len(synch_scores)) ** 0.1 if x >= 0 else -1
                for x in synch_scores
            ]
            synch_scores = [
                (float(x) - min(synch_scores))
                / (max(synch_scores) - min(synch_scores))
                if x >= 0
                else -1
                for x in synch_scores
            ]
        return synch_scores

    def get_keypoints_of_bodypart(
        self, pose, person_idx, keypoint_pair, bodypart_idx
    ):
        if self.synch_style in self.synch_styles_mirrored and person_idx == 1:
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

    def get_relevant_poses(self, poses):
        if len(poses) < 2:
            relevant_poses = None
        else:
            conf_vals = []
            for pose in poses:
                conf_vals.append(pose.confidence)
            if self.synch_style in SynchronyDetector.synch_styles_2persons:
                pax = 2
            elif self.synch_style == "allpax":
                pax = len(poses)
            else:
                pax = int(self.synch_style)
            relevant_poses_idx = (-conf_vals).argsort()[:pax]
            relevant_poses = [
                pose
                for idx, pose in enumerate(poses)
                if idx in relevant_poses_idx
            ]
        return relevant_poses

    def get_mirror_key(self, dict_key):
        org_idx = self.column_names.index(dict_key)
        mirror_idx = PoseEstimator.idx_mirror_pose_pair[org_idx]
        return self.column_names[mirror_idx]
