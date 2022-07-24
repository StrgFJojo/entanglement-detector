from scipy.spatial.distance import euclidean


class HeightCalculator:
    """
    Calculates the height of one or multiple poses/persons.
        Option 1: nose, neck, r-hip, r-knee, r-ankle
        Option 2: nose, neck, l-hip, l-knee, l-ankle
        Option 3: l-eye, neck, r-hip, r-knee, r-ankle
        Option 4: l-eye, neck, l-hip, l-knee, l-ankle
        Option 5: r-eye, neck, r-hip, r-knee, r-ankle
        Option 6: r-eye, neck, l-hip, l-knee, l-ankle
    """

    height_calculation_routes = [
        [[0, 1], [1, 8], [8, 9], [9, 10]],
        [[0, 1], [1, 11], [11, 12], [12, 13]],
        [[15, 1], [1, 8], [8, 9], [9, 10]],
        [[15, 1], [1, 11], [11, 12], [12, 13]],
        [[16, 1], [1, 8], [8, 9], [9, 10]],
        [[16, 1], [1, 11], [11, 12], [12, 13]],
    ]

    def calculate_heights(self, poses) -> None:
        heights = []
        if poses is not None and len(poses) > 1:
            for pose in poses:
                height = self.calculate_height(pose)
                heights.append(height)
        else:
            heights.append(-1)
        return heights

    def calculate_height(self, pose):
        for calc_route_idx, calc_route in enumerate(
            self.height_calculation_routes
        ):
            height = 0
            for keypoint_pair_idx, keypoint_pair in enumerate(calc_route):
                keypoint1 = pose.keypoints[keypoint_pair[0]]
                keypoint2 = pose.keypoints[keypoint_pair[1]]
                if -1 in keypoint1 or -1 in keypoint2:
                    break  # continue with next calc route
                height += euclidean(keypoint1, keypoint2)
                if keypoint_pair_idx == len(calc_route) - 1:
                    return height
            if calc_route_idx == len(self.height_calculation_routes) - 1:
                return -1
