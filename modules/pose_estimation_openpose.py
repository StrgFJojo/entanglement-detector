import numpy as np
from cv2 import cv2

protoFile = (
    "/Users/josephinevandelden/Documents/GitHub/learnopencv/"
    "OpenPose-Multi-Person/pose/coco/pose_deploy_linevec"
    ".prototxt "
)
weightsFile = "/Users/josephinevandelden/Downloads/pose_iter_440000.caffemodel"

nPoints = 18
# COCO Output Format
keypointsMapping = [
    "Nose",
    "Neck",
    "R-Sho",
    "R-Elb",
    "R-Wr",
    "L-Sho",
    "L-Elb",
    "L-Wr",
    "R-Hip",
    "R-Knee",
    "R-Ank",
    "L-Hip",
    "L-Knee",
    "L-Ank",
    "R-Eye",
    "L-Eye",
    "R-Ear",
    "L-Ear",
]

POSE_PAIRS = [
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
    [2, 17],
    [5, 16],
]

# 0: Neck, R-Sho (Mirror at index 1)
# 1: Neck, L-Sho (Mirror at index 0)
# 2: R-Sho, R-Elb (Mirror at index 4)
# 3: R-Elb, R-Wr (Mirror at index 5)
# 4: L-Sho, L-Elb (Mirror at index 2)
# 5: L-Elb, L-Wr (Mirror at index 3)
# 6: Neck, R-Hip (Mirror at index 9)
# 7: R-Hip, R-Knee (Mirror at index 10)
# 8: R-Knee, R-Ank (Mirror at index 11)
# 9: Neck, L-Hip (Mirror at index 6)
# 10: L-Hip, L-Knee (Mirror at index 7)
# 11: L-Knee, L-Ank (Mirror at index 8)
# 12: Neck, Nose (No mirror, or 12)
# 13: Nose, R-Eye (Mirror at index 15)
# 14: R-Eye, R-Ear (Mirror at index 16)
# 15: Nose, L-Eye (Mirror at index 13)
# 16: L-Eye, L-Ear (Mirror at index 14)
# 17: (R-Sho, L-Ear
# 18: L-Sho, R-Ear)

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

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices
# (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [
    [31, 32],
    [39, 40],
    [33, 34],
    [35, 36],
    [41, 42],
    [43, 44],
    [19, 20],
    [21, 22],
    [23, 24],
    [25, 26],
    [27, 28],
    [29, 30],
    [47, 48],
    [49, 50],
    [53, 54],
    [51, 52],
    [55, 56],
    [37, 38],
    [45, 46],
]

colors = [
    [0, 100, 255],
    [0, 100, 255],
    [0, 255, 255],
    [0, 100, 255],
    [0, 255, 255],
    [0, 100, 255],
    [0, 255, 0],
    [255, 200, 100],
    [255, 0, 255],
    [0, 255, 0],
    [255, 200, 100],
    [255, 0, 255],
    [0, 0, 255],
    [255, 0, 0],
    [200, 200, 0],
    [255, 0, 0],
    [200, 200, 0],
    [0, 0, 0],
]


# Find the Keypoints using Non Maximum Suppression on the Confidence Map
def getKeypoints(probMap, threshold):
    # def getKeypoints(probMap, threshold=0.1):

    mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)

    mapMask = np.uint8(mapSmooth > threshold)
    keypoints = []

    # find the blobs
    contours, _ = cv2.findContours(
        mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


# Find valid connections between the different joints of each person present
# finds keypoints pairs with min distance, and their direction
# complying with PAF heatmaps direction
def getValidPairs(output, frameWidth, frameHeight, detected_keypoints):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points
        # between the joints
        # Use the above formula to compute a score to mark the connection valid

        if nA != 0 and nB != 0:
            valid_pair = np.zeros((0, 3))
            for i in range(nA):
                max_j = -1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(
                        zip(
                            np.linspace(
                                candA[i][0], candB[j][0], num=n_interp_samples
                            ),
                            np.linspace(
                                candA[i][1], candB[j][1], num=n_interp_samples
                            ),
                        )
                    )
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append(
                            [
                                pafA[
                                    int(round(interp_coord[k][1])),
                                    int(round(interp_coord[k][0])),
                                ],
                                pafB[
                                    int(round(interp_coord[k][1])),
                                    int(round(interp_coord[k][0])),
                                ],
                            ]
                        )
                        # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores) / len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with
                    # PAF is higher then threshold -> Valid Pair
                    if (
                        len(np.where(paf_scores > paf_score_th)[0])
                        / n_interp_samples
                    ) > conf_th:
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(
                        valid_pair,
                        [[candA[i][3], candB[max_j][3], maxScore]],
                        axis=0,
                    )

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else:  # If no keypoints are detected
            # print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    # print(valid_pairs)
    return valid_pairs, invalid_pairs


# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
# It finds the person and index at which the joint should be added.
# This can be done since we have an id for each joint
def getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 19))

    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:, 0]
            partBs = valid_pairs[k][:, 1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += (
                        keypoints_list[partBs[i].astype(int), 2]
                        + valid_pairs[k][i][2]
                    )

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and
                    # the paf_score
                    row[-1] = (
                        sum(
                            keypoints_list[valid_pairs[k][i, :2].astype(int), 2]
                        )
                        + valid_pairs[k][i][2]
                    )
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints


def get_poses(frame, net, index, i):
    frameClone = frame.copy()
    # frameClone_forlater = frame.copy()
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    # Fix the input Height and get the width according to the Aspect Ratio
    # Default according to paper
    inHeight = 368
    inWidth = int((inHeight / frameHeight) * frameWidth)

    # preprocessing image, mean subtraction (combat illumination changes)
    # and scaling
    # image, scalefactor, size the net expects, mean
    # (vals subtracted from every channel of the image)
    # OpenCV assumes images are in BGR channel order;
    # however `mean` value assumes we are using RGB order
    # Swap not necessary as mean set to 0
    inpBlob = cv2.dnn.blobFromImage(
        frame,
        1.0 / 255,
        (inWidth, inHeight),
        (0, 0, 0),
        swapRB=False,
        crop=False,
    )

    net.setInput(inpBlob)
    # Output contains confidence maps and PAFs
    # grayscale image which has a high value at locations
    # where the likelihood of a certain body part is high
    # 18 point model ->  first 19 matrices of the output correspond
    # to the confidence maps
    # used to find the keypoints
    # matrices 20 to 57 are PAF part affinity maps,
    # encoding the degree of association between parts (keypoints)
    # Affinity Maps are used to get the valid connections
    # between the keypoints.
    output = net.forward()
    # print("Time Taken = {}".format(time.time() - t))

    # Slice a probability map (for e.g the nose)
    # from the output for a specific keypoint
    detected_keypoints = []
    keypoints_list = np.zeros((0, 3))
    # start ids with zero
    keypoint_id = 0
    # increased from 0.1 to 0.5 to get rid of background people
    threshold = 0.5

    # for each keypoint
    for part in range(nPoints):
        probMap = output[0, part, :, :]
        # resize the output to input size
        probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))

        # use method to detect keypoints
        # for a single person, it is very easy to find the location of
        # each keypoint:
        # just find maximum of the confidence map
        # for multi-person scenario, we can’t do this
        # keypoints returns x, y coordinates and prob for
        # each person and a specific keypoint
        # TODO: use low probs (around 0.11) of background people
        # vs high props of foreground
        # people (around 0.9) to get rid of background people
        # -> increase threshold
        keypoints = getKeypoints(probMap, threshold)
        # print("Keypoints - {} : {}".format(keypointsMapping[part],
        # keypoints))

        keypoints_with_id = []
        for i in range(
            len(keypoints)
        ):  # len of keypoints list corresponds to number of people
            keypoints_with_id.append(
                keypoints[i] + (keypoint_id,)
            )  # xy coordinates and prob, then add id
            keypoints_list = np.vstack(
                [keypoints_list, keypoints[i]]
            )  # stacks all keypoints with id of all persons
            keypoint_id += 1
        # Rows -> Keypoints (18), Columns -> Num of people (2)
        # Some rows can be empty if keypoint was not detected
        # Per person and per keypoint, there are 4 vals: xy coordinates,
        # probability, id
        # Empty rows do not have an id
        detected_keypoints.append(keypoints_with_id)

    # finds keypoints pairs with min distance,
    # and their direction complying with PAF heatmaps direction
    valid_pairs, invalid_pairs = getValidPairs(
        output, frameWidth, frameHeight, detected_keypoints
    )

    # assemble pairs that share the same part detection candidates
    # into full-body poses of multiple people
    # for each person, the keypoint id for the 18 keypoints is given,
    # followed by the summed score
    # i.e. for person 1, the first keypoint, which is the nose,
    # has the keypoint id 0
    # coordinates of the keypoint can be found in detected_keypoints
    # list or keypoints_list
    # detected_keypoints stacks the keypoints with its coordinates,
    # probabilities and ids
    # keypoints_list similar to detected_keypoints but without id
    # detected_keypoints and keypoints_list indices equal keypoint id
    personwiseKeypoints = getPersonwiseKeypoints(
        valid_pairs, invalid_pairs, keypoints_list
    )

    return frameClone, keypoints_list, detected_keypoints, personwiseKeypoints
