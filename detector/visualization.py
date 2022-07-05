import textwrap

import numpy as np
import cv2
from statistics import mean
from PIL import Image, ImageDraw

idx_mirror_pose_pair = [1, 0, 4, 5, 2, 3, 9, 10, 11, 6, 7, 8, 12, 15, 16, 13, 14]


def round_corner(radius, fill):
    """Draw a round corner"""
    corner = Image.new('RGB', (radius, radius), (0, 0, 0, 0))
    draw = ImageDraw.Draw(corner)
    draw.pieslice((0, 0, radius * 2, radius * 2), 180, 270, fill=fill)
    return corner


def round_rectangle(size, radius, fill):
    """Draw a rounded rectangle"""
    width, height = size
    rectangle = Image.new('RGB', size, fill)
    corner = round_corner(radius, fill)
    rectangle.paste(corner, (0, 0))
    rectangle.paste(corner.rotate(90), (0, height - radius))  # Rotate the corner and paste it
    rectangle.paste(corner.rotate(180), (width - radius, height - radius))
    rectangle.paste(corner.rotate(270), (width - radius, 0))
    return rectangle


def overlay_dashboard(text, font, font_scale, font_thickness):
    # get boundary of this text
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    background = round_rectangle((textsize[0], textsize[1] * 2), 10, "white")
    dash = np.array(background)
    # Convert RGB to BGR
    dash = dash[:, :, ::-1].copy()
    # get coords based on boundary
    textX = (dash.shape[1] - textsize[0]) / 2
    textY = (dash.shape[0] + textsize[1]) / 2

    # add text centered on image
    cv2.putText(dash, text, (int(textX), int(textY)), font, 1, (0, 0, 0), 2)

    return dash


def full_skeleton_w_synchdegree(personwiseKeypoints, POSE_PAIRS, keypoints_list, synch_degree, frameClone):
    for l in range(17):
        # go over each person
        for m in range(len(personwiseKeypoints)):
            # this may plot skeletons looking like one person but actually being two or more
            # personwiseKeypoints[person][pulling pair tuple from list]
            # eg. [0][POSE_PAIRS[0]] -> [0][1,2] first person, first pose_pair which is [1,2] or (neck,r-shoulder)
            # personwiseKeypoints[0][1,2] = [2,4] -> keypoints for (neck,r-shoulder) of 1st person have ids 2 and 4
            index2 = personwiseKeypoints[m][np.array(POSE_PAIRS[l])]
            if -1 in index2:
                # one of the pairs' keypoints is not available
                continue
            # look up coordinates for keypoints
            B = np.int32(keypoints_list[index2.astype(int), 0])
            A = np.int32(keypoints_list[index2.astype(int), 1])

            if synch_degree[l] == -1:
                color = [0, 0, 0]
            else:
                color = [0,
                         min(255, 2 * 255 * (synch_degree[l])),
                         min(255, 2 * 255 * (1 - synch_degree[l]))]
            cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), color, 2, cv2.LINE_AA)
    return frameClone


def partly_overlay(personwiseKeypoints, POSE_PAIRS, keypoints_list, synch_degree, frameClone):
    overlay = frameClone.copy()
    synch_temp = []
    for l in [2, 3, 4, 5, 10, 11, 7, 8]:
        if synch_degree[l] != -1:
            synch_temp.append(synch_degree[l])
        # go over each person
        """
        for m in range(len(personwiseKeypoints)):
            # this may plot skeletons looking like one person but actually being two or more
            # personwiseKeypoints[person][pulling pair tuple from list]
            # eg. [0][POSE_PAIRS[0]] -> [0][1,2] first person, first pose_pair which is [1,2] or (neck,r-shoulder)
            # personwiseKeypoints[0][1,2] = [2,4] -> keypoints for (neck,r-shoulder) of 1st person have ids 2 and 4
            index2 = personwiseKeypoints[m][np.array(POSE_PAIRS[l])]
            if -1 in index2:
                # one of the pairs' keypoints is not available
                continue
            # look up coordinates for keypoints
            B = np.int32(keypoints_list[index2.astype(int), 0])
            A = np.int32(keypoints_list[index2.astype(int), 1])

            if synch_degree[l] == -1:
                color = [0, 0, 0]
            else:
                color = [0,
                         min(255, 2 * 255 * (synch_degree[l])),
                         min(255, 2 * 255 * (1 - synch_degree[l]))]
            cv2.line(overlay, (B[0], A[0]), (B[1], A[1]), color, 3, cv2.LINE_AA)

    result = cv2.addWeighted(overlay, 0.70, frameClone, 1 - 0.70, 0)
    """
    if len(synch_temp) == 0:
        dash = overlay_dashboard("Avg limb synch: na", cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    else:
        synch_mean = mean(synch_temp)
        dash = overlay_dashboard(str("Avg limb synch: %.2f" % synch_mean), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

    x_offset = y_offset = 50
    overlay[y_offset:y_offset + dash.shape[0], x_offset:x_offset + dash.shape[1]] = dash
    # cv2.putText(result, str(synch_mean), (10,50),cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255, 0, 0))
    return overlay


def partly_overlay_multipax(personwiseKeypoints, indices_of_two_highest_scores, POSE_PAIRS, keypoints_list,
                            synch_degree, distance, frameClone, synch_style):
    overlay = frameClone.copy()
    synch_temp = []
    if synch_style == '2pax_90' or synch_style == '2pax_180':
        for l in [2, 3, 4, 5, 10, 11, 7, 8]:
            if synch_degree[l] != -1:
                synch_temp.append(synch_degree[l])
            # go over each person
            """
            for m in range(len(indices_of_two_highest_scores)):
                # this may plot skeletons looking like one person but actually being two or more
                # personwiseKeypoints[person][pulling pair tuple from list]
                # eg. [0][POSE_PAIRS[0]] -> [0][1,2] first person, first pose_pair which is [1,2] or (neck,r-shoulder)
                # personwiseKeypoints[0][1,2] = [2,4] -> keypoints for (neck,r-shoulder) of 1st person have ids 2 and 4
                index2 = personwiseKeypoints[indices_of_two_highest_scores[m]][np.array(POSE_PAIRS[l])]
                if -1 in index2:
                    # one of the pairs' keypoints is not available
                    continue
                # look up coordinates for keypoints
                B = np.int32(keypoints_list[index2.astype(int), 0])
                A = np.int32(keypoints_list[index2.astype(int), 1])

                if synch_degree[l] == -1:
                    color = [0, 0, 0]
                else:
                    color = [0,
                             min(255, 2 * 255 * (synch_degree[l])),
                             min(255, 2 * 255 * (1 - synch_degree[l]))]
                cv2.line(overlay, (B[0], A[0]), (B[1], A[1]), color, 1, cv2.LINE_AA)

        result = cv2.addWeighted(overlay, 0.70, frameClone, 1 - 0.70, 0)
        """
        if len(synch_temp) == 0 and distance == -1:
            dash = overlay_dashboard("Avg limb synch: na --- Dist: na", cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        elif len(synch_temp) == 0 and distance != -1:
            dash = overlay_dashboard(str("Avg limb synch: na --- Dist: %.2f px" % distance), cv2.FONT_HERSHEY_SIMPLEX,
                                     0.4, 1)
        elif len(synch_temp) != 0 and distance == -1:
            synch_mean = mean(synch_temp)
            dash = overlay_dashboard(str("Avg limb synch: %.2f --- Dist: na" % synch_mean), cv2.FONT_HERSHEY_SIMPLEX,
                                     0.4, 1)
        else:
            synch_mean = mean(synch_temp)
            dash = overlay_dashboard(str("Avg limb synch: %.2f --- Dist: %.2f px" % (synch_mean, distance)),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        x_offset = y_offset = 10
        overlay[y_offset:y_offset + dash.shape[0], x_offset:x_offset + dash.shape[1]] = dash
        # cv2.putText(result, str(synch_mean), (10,50),cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255, 0, 0))
    else:
        for l in [2, 3, 4, 5, 10, 11, 7, 8]:
            if synch_degree[l] != -1:
                synch_temp.append(synch_degree[l])
            # go over each person
            """
            for m in range(len(personwiseKeypoints)):
                # this may plot skeletons looking like one person but actually being two or more
                # personwiseKeypoints[person][pulling pair tuple from list]
                # eg. [0][POSE_PAIRS[0]] -> [0][1,2] first person, first pose_pair which is [1,2] or (neck,r-shoulder)
                # personwiseKeypoints[0][1,2] = [2,4] -> keypoints for (neck,r-shoulder) of 1st person have ids 2 and 4
                index2 = personwiseKeypoints[m][np.array(POSE_PAIRS[l])]
                if -1 in index2:
                    # one of the pairs' keypoints is not available
                    continue
                # look up coordinates for keypoints
                B = np.int32(keypoints_list[index2.astype(int), 0])
                A = np.int32(keypoints_list[index2.astype(int), 1])

                if synch_degree[l] == -1:
                    color = [0, 0, 0]
                else:
                    color = [0,
                             min(255, 2 * 255 * (synch_degree[l])),
                             min(255, 2 * 255 * (1 - synch_degree[l]))]
                cv2.line(overlay, (B[0], A[0]), (B[1], A[1]), color, 3, cv2.LINE_AA)

        result = cv2.addWeighted(overlay, 0.70, frameClone, 1 - 0.70, 0)
        """
        if len(synch_temp) == 0 and distance == -1:
            dash = overlay_dashboard("Avg limb synch: na --- Dist: na", cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        elif len(synch_temp) == 0 and distance != -1:
            dash = overlay_dashboard(str("Avg limb synch: na --- Dist: %.2f px" % distance), cv2.FONT_HERSHEY_SIMPLEX,
                                     0.4, 1)
        elif len(synch_temp) != 0 and distance == -1:
            synch_mean = mean(synch_temp)
            dash = overlay_dashboard(str("Avg limb synch: %.2f --- Dist: na" % synch_mean), cv2.FONT_HERSHEY_SIMPLEX,
                                     0.4, 1)
        else:
            synch_mean = mean(synch_temp)
            dash = overlay_dashboard(str("Avg limb synch: %.2f --- Dist: %.2f px" % (synch_mean, distance)),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        x_offset = y_offset = 50
        overlay[y_offset:y_offset + dash.shape[0], x_offset:x_offset + dash.shape[1]] = dash
        # cv2.putText(result, str(synch_mean), (10,50),cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255, 0, 0))
    return overlay


def draw_text(img, text,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=3,
              font_thickness=2,
              text_color=(0, 255, 0),
              text_color_bg=(0, 0, 0)
              ):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size


def partly_overlay_final(personwiseKeypoints, person_indices, POSE_PAIRS, keypoints_list, synch_degree, distance,
                         frameClone, synch_style):
    overlay = frameClone.copy()

    if len(personwiseKeypoints) == 0:
        dash = overlay_dashboard("Avg limb synch: na --- Dist: na", cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        x_offset = y_offset = 10
        overlay[y_offset:y_offset + dash.shape[0], x_offset:x_offset + dash.shape[1]] = dash
        return overlay

    for l in range(17):
        for m in range(len(person_indices)):
            idx = personwiseKeypoints[person_indices[m]][np.array(POSE_PAIRS[l])]
            if -1 in idx:
                # one of the pairs' keypoints is not available
                continue
            # look up coordinates for keypoints
            B = np.int32(keypoints_list[idx.astype(int), 0])
            A = np.int32(keypoints_list[idx.astype(int), 1])

            if m in person_indices and synch_degree[l] != -1 and not (
                    synch_style in ['2pax_90_mirrored', '2pax_180_mirrored'] and m == 1):
                color = [0,
                         min(255, 2 * 255 * (synch_degree[l])),
                         min(255, 2 * 255 * (1 - synch_degree[l]))]
            elif synch_style in ['2pax_90_mirrored', '2pax_180_mirrored'] and m == 1 and synch_degree[l] != -1:
                color = [0,
                         min(255, 2 * 255 * (synch_degree[idx_mirror_pose_pair[l]])),
                         min(255, 2 * 255 * (1 - synch_degree[idx_mirror_pose_pair[l]]))]
            else:
                color = [0, 0, 0]

            cv2.line(overlay, (B[0], A[0]), (B[1], A[1]), color, 1, cv2.LINE_AA)

    result = cv2.addWeighted(overlay, 0.70, frameClone, 1 - 0.70, 0)

    synch_temp_limbs = [x for x in np.array(synch_degree)[[2, 3, 4, 5, 10, 11, 7, 8]] if x != -1]
    synch_temp_face = [x for x in np.array(synch_degree)[[12, 13, 14, 15, 16]] if x != -1]

    synch_mean_limbs = mean(synch_temp_limbs) if len(synch_temp_limbs) != 0 else False
    synch_mean_face = mean(synch_temp_face) if len(synch_temp_face) != 0 else False

    #overlay_text = f"Limb synch: {synch_mean_limbs or np.nan:.1f}; Face synch: {synch_mean_face or np.nan:.1f}; Dist: {np.nan if distance == -1 else distance:.1f} px."
    #wrapper = textwrap.TextWrapper(width=overlay.shape[1])
    #overlay_text = wrapper.fill(text=overlay_text)

    overlay_text = f"Limb synch: {synch_mean_limbs or np.nan:.1f}; Dist: {np.nan if distance == -1 else distance:.1f} px."
    dash = overlay_dashboard(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

    x_offset = y_offset = 10
    overlay[y_offset:y_offset + dash.shape[0], x_offset:x_offset + dash.shape[1]] = dash
    # cv2.putText(result, str(synch_mean), (10,50),cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255, 0, 0))

    return overlay
