import numpy as np
import math
from statistics import mean

idx_mirror_pose_pair = [1, 0, 4, 5, 2, 3, 9, 10, 11, 6, 7, 8, 12, 15, 16, 13, 14]


def get_synchrony(indices_of_two_highest_scores, personwiseKeypoints, POSE_PAIRS, keypoints_list,
                  synch_style='2pax_90'):
    # Synchrony

    synch_degree = -1 * np.ones(17)

    if synch_style in ['2pax_90', '2pax_180', '2pax_90_mirrored', '2pax_180_mirrored']:

        for n in range(17):
            unit_vectors = -1 * np.ones((len(indices_of_two_highest_scores), 2))
            for i in range(len(indices_of_two_highest_scores)):  # for each of the two main people
                # personwiseKeypoints[person][pulling pair tuple from list]
                # eg. [0][POSE_PAIRS[0]] -> [0][1,2] first person, first pose_pair which is [1,2] or (neck,r-shoulder)
                # personwiseKeypoints[0][1,2] = [2,4] -> keypoints for (neck,r-shoulder) of 1st person have ids 2 and 4
                if synch_style in ['2pax_90_mirrored', '2pax_180_mirrored'] and i == 1:
                    index = personwiseKeypoints[indices_of_two_highest_scores[i]][np.array(POSE_PAIRS[idx_mirror_pose_pair[n]])]
                else:
                    index = personwiseKeypoints[indices_of_two_highest_scores[i]][np.array(POSE_PAIRS[n])]

                if -1 in index:
                    # one of the pairs' keypoints is not available
                    unit_vectors[i] = [-1, -1]
                    continue
                # look up coordinates for keypoints
                x_vals = np.int32(keypoints_list[index.astype(int), 0])  # for both key points of pose pair
                y_vals = np.int32(keypoints_list[index.astype(int), 1])  # for both key points of pose pair

                # get vector
                distance = [x_vals[1] - x_vals[0], y_vals[1] - y_vals[0]]
                norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2)
                unit_vector = [distance[0] / norm, distance[1] / norm]  # divide vector by its norm

                unit_vectors[i] = unit_vector

            if (-1 in unit_vectors[0]) or (-1 in unit_vectors[1]):
                synch_degree[n] = -1
            else:
                dot_product = np.dot(unit_vectors[0], unit_vectors[1])
                rad = np.arccos(dot_product)  # in radians
                deg = np.rad2deg(rad)  # in degree
                if synch_style in ['2pax_90', '2pax_90_mirrored']:
                    if deg <= 90:
                        val = (1 + ((0 - 1) / (90 - 0)) * (deg - 0))
                    else:
                        val = 0 + ((1 - 0) / (180 - 90)) * (deg - 90)
                else:  # synch_style == '2pax_180'
                    val = deg / 180

                synch_degree[n] = val  # 1 -> perfect synch, 0 -> no synch

        assert isinstance(synch_degree, object)
    elif synch_style == 'allpax':

        for n in range(17):
            unit_vectors = -1 * np.ones((len(personwiseKeypoints), 2))
            for i in range(len(personwiseKeypoints)):  # for each of the two main people
                # personwiseKeypoints[person][pulling pair tuple from list]
                # eg. [0][POSE_PAIRS[0]] -> [0][1,2] first person, first pose_pair which is [1,2] or (neck,r-shoulder)
                # personwiseKeypoints[0][1,2] = [2,4] -> keypoints for (neck,r-shoulder) of 1st person have ids 2 and 4
                index = personwiseKeypoints[i][np.array(POSE_PAIRS[n])]
                if -1 in index:
                    # one of the pairs' keypoints is not available
                    unit_vectors[i] = [-1, -1]
                    continue
                # look up coordinates for keypoints
                x_vals = np.int32(keypoints_list[index.astype(int), 0])  # for both key points of pose pair
                y_vals = np.int32(keypoints_list[index.astype(int), 1])  # for both key points of pose pair

                # get vector
                distance = [x_vals[1] - x_vals[0], y_vals[1] - y_vals[0]]
                norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2)
                unit_vector = [distance[0] / norm, distance[1] / norm]  # divide vector by its norm


                unit_vectors[i] = unit_vector
            unit_vectors = [x for x in unit_vectors if -1 not in x]
            vector_sum = [0, 0]
            for vecs in unit_vectors:
                vector_sum += vecs
            if len(unit_vectors) > 0:
                avg_vec = [x/len(unit_vectors) for x in vector_sum]
                dif_vectors_sum = 0
                for j in range(len(unit_vectors)):
                    dif_vectors_sum += abs(unit_vectors[j] - avg_vec)
                synch_degree[n] = mean(dif_vectors_sum)
            else:
                synch_degree[n] = -1
        synch_degree = [(float(x) / len(synch_degree))**0.5 if x >= 0 else -1 for x in synch_degree]
        synch_degree = [(float(x) - min(synch_degree)) / (max(synch_degree) - min(synch_degree)) if x >= 0 else -1 for x
                        in synch_degree]

    return synch_degree


def get_synchrony_flexpax(pose_entries, person_indices, POSE_PAIRS, all_keypoints, synch_style):
    synch_degree = -1 * np.ones(17)
    for n in range(17):
        unit_vectors = -1 * np.ones((len(person_indices), 2))
        for i in range(len(person_indices)):  # for each of the persons selected
            # personwiseKeypoints[person][pulling pair tuple from list]
            # eg. [0][POSE_PAIRS[0]] -> [0][1,2] first person, first pose_pair which is [1,2] or (neck,r-shoulder)
            # personwiseKeypoints[0][1,2] = [2,4] -> keypoints for (neck,r-shoulder) of 1st person have ids 2 and 4
            index = pose_entries[person_indices[i]][np.array(POSE_PAIRS[n])]
            if -1 in index:
                # one of the pairs' keypoints is not available
                unit_vectors[i] = [-1, -1]
                continue
            # look up coordinates for keypoints
            x_vals = np.int32(all_keypoints[index.astype(int), 0])  # for both key points of pose pair
            y_vals = np.int32(all_keypoints[index.astype(int), 1])  # for both key points of pose pair

            # get vector
            distance = [x_vals[1] - x_vals[0], y_vals[1] - y_vals[0]]
            norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2)
            unit_vector = [distance[0] / norm, distance[1] / norm]  # divide vector by its norm
            unit_vectors[i] = unit_vector

        if synch_style in ['2pax_90', '2pax_180', '2pax_90_mirrored', '2pax_180_mirrored']:
            if (-1 in unit_vectors[0]) or (-1 in unit_vectors[1]):
                synch_degree[n] = -1
            else:
                dot_product = np.dot(unit_vectors[0], unit_vectors[1])
                rad = np.arccos(dot_product)  # in radians
                deg = np.rad2deg(rad)  # in degree
                if synch_style in ['2pax_90', '2pax_90_mirrored']:
                    if deg <= 90:
                        val = (1 + ((0 - 1) / (90 - 0)) * (deg - 0))
                    else:
                        val = 0 + ((1 - 0) / (180 - 90)) * (deg - 90)
                else:  # synch_style == '2pax_180'
                    val = deg / 180
                synch_degree[n] = val  # 1 -> perfect synch, 0 -> no synch
        else:
            unit_vectors = [x for x in unit_vectors if -1 not in x]
            vector_sum = [0, 0]
            for vec in unit_vectors:
                vector_sum += vec
            if len(unit_vectors) > 1:
                avg_vec = [x/len(unit_vectors) for x in vector_sum]
                dif_vectors_sum = 0
                for j in range(len(unit_vectors)):
                    dif_vectors_sum += abs(unit_vectors[j] - avg_vec)
                synch_degree[n] = mean(dif_vectors_sum)
            else:
                synch_degree[n] = -1

    if synch_style not in ['2pax_90', '2pax_180', '2pax_90_mirrored', '2pax_180_mirrored']:
        synch_degree = [(float(x)/len(synch_degree))**0.5 if x >= 0 else -1 for x in synch_degree]
        synch_degree = [(float(x)-min(synch_degree))/(max(synch_degree)-min(synch_degree)) if x >= 0 else -1 for x in synch_degree]

    return synch_degree
