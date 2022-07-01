import numpy as np
from scipy.spatial import distance as dist


def get_height(indices_of_two_highest_scores, personwiseKeypoints, keypoints_list=None):

    pose_pairs_height_options = [[[0,1], [1,8], [8,9], [9,10]], # option 1: nose, neck, r-hip, r-knee, r-ankle
                                [[0,1], [1,11], [11,12], [12,13]], # option 2: nose, neck, l-hip, l-knee, l-ankle
                                [[15,1], [1,8], [8,9], [9,10]], # option 3: l-eye, neck, r-hip, r-knee, r-ankle
                                [[15,1], [1,11], [11,12], [12,13]], # option 4: # l-eye, neck, l-hip, l-knee, l-ankle
                                [[16,1], [1,8], [8,9], [9,10]], # option 5: r-eye, neck, r-hip, r-knee, r-ankle
                                [[16,1], [1,11], [11,12], [12,13]]] # option 6: r-eye, neck, l-hip, l-knee, l-ankle

    pose_pairs_height_options_names = ["nose, neck, r-hip, r-knee, r-ankle",
                                      "nose, neck, l-hip, l-knee, l-ankle",
                                      "l-eye, neck, r-hip, r-knee, r-ankle",
                                      "l-eye, neck, l-hip, l-knee, l-ankle",
                                      "r-eye, neck, r-hip, r-knee, r-ankle",
                                      "r-eye, neck, l-hip, l-knee, l-ankle"]
    flag = False
    done = False
    height = 0
    heights = -1 * np.ones(len(indices_of_two_highest_scores))

    for i in range(len(indices_of_two_highest_scores)): # for each of the two main people
        for j in range(len(pose_pairs_height_options)): # for each of the options of calculation

            if (done == True) & (i < len(indices_of_two_highest_scores)-1):
                #print("-> Break out from loops and continue with next person\n")
                height = 0 # reset height
                done = False
                break
            elif (done == True) & (i == len(indices_of_two_highest_scores)-1):
                #print ("\n-> Fully done.\nPerson index, height:\n%s" % heights)
                break

            for k in range(len(pose_pairs_height_options[0])): # for each pair of keypoints relevant for total height

                index_of_keypoints = personwiseKeypoints[indices_of_two_highest_scores[i].astype(int)][np.array(pose_pairs_height_options[j][k])]

                if -1 in index_of_keypoints:  # one of the pairs' keypoints is not available
                    # break out of the inner loop and continue with next calculation options
                    flag = True
                    break

                keypoint1 = [keypoints_list[index_of_keypoints[0].astype(int), 0],
                            keypoints_list[index_of_keypoints[0].astype(int), 1]]

                keypoint2 = [keypoints_list[index_of_keypoints[1].astype(int), 0],
                            keypoints_list[index_of_keypoints[1].astype(int), 1]]

                height += dist.euclidean(keypoint1, keypoint2)

                # if distances were calculated for all pose pairs of the current option
                # no further options have to be considered
                if k == len(pose_pairs_height_options[0])-1:
                    done = True
                    heights[i] = height

    return heights


def get_height_multipax(personwiseKeypoints, keypoints_list, person_indices):

    pose_pairs_height_options = [[[0,1], [1,8], [8,9], [9,10]], # option 1: nose, neck, r-hip, r-knee, r-ankle
                                [[0,1], [1,11], [11,12], [12,13]], # option 2: nose, neck, l-hip, l-knee, l-ankle
                                [[15,1], [1,8], [8,9], [9,10]], # option 3: l-eye, neck, r-hip, r-knee, r-ankle
                                [[15,1], [1,11], [11,12], [12,13]], # option 4: # l-eye, neck, l-hip, l-knee, l-ankle
                                [[16,1], [1,8], [8,9], [9,10]], # option 5: r-eye, neck, r-hip, r-knee, r-ankle
                                [[16,1], [1,11], [11,12], [12,13]]] # option 6: r-eye, neck, l-hip, l-knee, l-ankle

    pose_pairs_height_options_names = ["nose, neck, r-hip, r-knee, r-ankle",
                                      "nose, neck, l-hip, l-knee, l-ankle",
                                      "l-eye, neck, r-hip, r-knee, r-ankle",
                                      "l-eye, neck, l-hip, l-knee, l-ankle",
                                      "r-eye, neck, r-hip, r-knee, r-ankle",
                                      "r-eye, neck, l-hip, l-knee, l-ankle"]
    flag = False
    done = False
    height = 0
    heights = -1 * np.ones(len(person_indices))

    for i in range(len(person_indices)): # for each of the two main people
        for j in range(len(pose_pairs_height_options)): # for each of the options of calculation

            if (done == True) & (i < len(person_indices)-1):
                #print("-> Break out from loops and continue with next person\n")
                height = 0 # reset height
                done = False
                break
            elif (done == True) & (i == len(person_indices)-1):
                #print ("\n-> Fully done.\nPerson index, height:\n%s" % heights)
                break

            for k in range(len(pose_pairs_height_options[0])): # for each pair of keypoints relevant for total height

                index_of_keypoints = personwiseKeypoints[person_indices[i]][np.array(pose_pairs_height_options[j][k])]

                if -1 in index_of_keypoints:  # one of the pairs' keypoints is not available
                    # break out of the inner loop and continue with next calculation options
                    flag = True
                    break

                keypoint1 = [keypoints_list[index_of_keypoints[0].astype(int), 0],
                            keypoints_list[index_of_keypoints[0].astype(int), 1]]

                keypoint2 = [keypoints_list[index_of_keypoints[1].astype(int), 0],
                            keypoints_list[index_of_keypoints[1].astype(int), 1]]

                height += dist.euclidean(keypoint1, keypoint2)

                # if distances were calculated for all pose pairs of the current option
                # no further options have to be considered
                if k == len(pose_pairs_height_options[0])-1:
                    done = True
                    heights[i] = height

    return heights