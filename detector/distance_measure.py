import numpy as np
from scipy.spatial import distance as dist


def get_distance(
    personwiseKeypoints, indices_of_two_highest_scores, keypoints_list
):
    # print("persons", len(indices_of_two_highest_scores))
    # center points of their bodies
    center_points = np.zeros((len(indices_of_two_highest_scores), 2))

    # for both skaters
    for j in range(len(indices_of_two_highest_scores)):

        # get coordinates of right hip
        keypoint_id_rhip = personwiseKeypoints[
            indices_of_two_highest_scores[j]
        ][
            8
        ]  # id is also index
        if keypoint_id_rhip == -1:
            x_rhip = -1
            y_rhip = -1
        else:
            x_rhip = keypoints_list[keypoint_id_rhip.astype(int), 0]
            y_rhip = keypoints_list[keypoint_id_rhip.astype(int), 1]

        # get coordinates of left hip
        keypoint_id_lhip = personwiseKeypoints[
            indices_of_two_highest_scores[j]
        ][
            11
        ]  # id is also index
        if keypoint_id_lhip == -1:
            x_lhip = -1
            y_lhip = -1
        else:
            x_lhip = keypoints_list[keypoint_id_lhip.astype(int), 0]
            y_lhip = keypoints_list[keypoint_id_lhip.astype(int), 1]

        # calculate coordinates for center of left and right hip
        # x coordinate
        if (x_rhip != -1) & (x_lhip != -1):
            x_midhip = (x_rhip + x_lhip) / 2
        elif (x_rhip == -1) & (x_lhip != -1):
            x_midhip = x_lhip
        elif (x_rhip != -1) & (x_lhip == -1):
            x_midhip = x_rhip
        elif (x_rhip == -1) & (x_lhip == -1):
            x_midhip = -1
            # print("Center of skater could not be calculated")

        # y coordinate
        if (y_rhip != -1) & (y_lhip != -1):
            y_midhip = (y_rhip + y_lhip) / 2
        elif (y_rhip == -1) & (y_lhip != -1):
            y_midhip = y_lhip
        elif (y_rhip != -1) & (y_lhip == -1):
            y_midhip = y_rhip
        elif (y_rhip == -1) & (y_lhip == -1):
            y_midhip = -1
            # print("Center of skater could not be calculated")

        center_points[j] = [int(x_midhip), int(y_midhip)]
    if -1 in center_points[0] or -1 in center_points[1]:
        distance_midhip = -1
    else:
        distance_midhip = dist.euclidean(center_points[0], center_points[1])

    return distance_midhip


def get_distance_multipax(personwiseKeypoints, keypoints_list, person_indices):
    # print("persons", len(indices_of_two_highest_scores))
    # center points of their bodies
    center_points = np.zeros((len(person_indices), 2))
    distance_midhip = -1

    # for both skaters
    for j in range(len(person_indices)):

        # get coordinates of right hip
        keypoint_id_rhip = personwiseKeypoints[person_indices[j]][
            8
        ]  # id is also index
        if keypoint_id_rhip == -1:
            x_rhip = -1
            y_rhip = -1
        else:
            x_rhip = keypoints_list[keypoint_id_rhip.astype(int), 0]
            y_rhip = keypoints_list[keypoint_id_rhip.astype(int), 1]

        # get coordinates of left hip
        keypoint_id_lhip = personwiseKeypoints[person_indices[j]][
            11
        ]  # id is also index
        if keypoint_id_lhip == -1:
            x_lhip = -1
            y_lhip = -1
        else:
            x_lhip = keypoints_list[keypoint_id_lhip.astype(int), 0]
            y_lhip = keypoints_list[keypoint_id_lhip.astype(int), 1]

        # calculate coordinates for center of left and right hip
        # x coordinate
        if (x_rhip != -1) & (x_lhip != -1):
            x_midhip = (x_rhip + x_lhip) / 2
        elif (x_rhip == -1) & (x_lhip != -1):
            x_midhip = x_lhip
        elif (x_rhip != -1) & (x_lhip == -1):
            x_midhip = x_rhip
        elif (x_rhip == -1) & (x_lhip == -1):
            x_midhip = -1
            # print("Center of skater could not be calculated")

        # y coordinate
        if (y_rhip != -1) & (y_lhip != -1):
            y_midhip = (y_rhip + y_lhip) / 2
        elif (y_rhip == -1) & (y_lhip != -1):
            y_midhip = y_lhip
        elif (y_rhip != -1) & (y_lhip == -1):
            y_midhip = y_rhip
        elif (y_rhip == -1) & (y_lhip == -1):
            y_midhip = -1
            # print("Center of skater could not be calculated")

        center_points[j] = [int(x_midhip), int(y_midhip)]
    center_points = [x for x in center_points if -1 not in x]
    if len(center_points) > 1:
        center_points = sorted(
            center_points, key=lambda x: x[0]
        )  # sorted, ascending x coordinates
        distance_midhip = 0
        for k in range(len(center_points) - 1):
            distance_midhip += abs(
                dist.euclidean(center_points[k], center_points[k + 1])
            )
    else:
        distance_midhip = -1

    return distance_midhip
