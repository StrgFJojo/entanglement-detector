from scipy.spatial.distance import euclidean


class DistanceCalculator:
    """
    Calculates the distance between 2 or more bodies based on the pose list.
    Distance is measured between the bodies' centers of gravity, which
    is approximated by taking the center point between the left and right hip
    keypoints.
    """

    keypoint_idx_lhip = 11
    keypoint_idx_rhip = 8

    def __init__(self):
        self.normalized_distance = []

    def calculate_distance(self, poses):
        if poses is None:
            distance = -1
        else:
            center_points = []

            for pose in poses:
                keypoint_lhip = pose.keypoints[self.keypoint_idx_lhip]
                keypoint_rhip = pose.keypoints[self.keypoint_idx_rhip]

                if -1 in keypoint_lhip or -1 in keypoint_rhip:
                    center_points.append([-1, -1])
                else:
                    center_point = [
                        (keypoint_lhip[0] + keypoint_rhip[0]) / 2,
                        (keypoint_lhip[1] + keypoint_rhip[1]) / 2,
                    ]
                    center_points.append(center_point)

            center_points = [
                center_point
                for center_point in center_points
                if -1 not in center_point
            ]

            if len(center_points) > 1:
                # sort; ascending x coordinates
                center_points = sorted(center_points, key=lambda x: x[0])
                distance_temp = 0
                for i in range(len(center_points) - 1):
                    distance_temp += euclidean(
                        center_points[i], center_points[i + 1]
                    )
                distance = distance_temp
            else:
                distance = -1
        return distance

    def normalize_distance(self, distance, heights):
        if -1 in heights or distance == -1:
            normalized_distance = dict({"normalized_distance": -1})
        else:
            normalized_distance = dict(
                {"normalized_distance": (distance / sum(heights))}
            )
        self.normalized_distance.append(normalized_distance)
        return normalized_distance
