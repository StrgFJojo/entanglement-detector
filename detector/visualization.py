from statistics import mean

import cv2
import numpy as np
from PIL import Image, ImageDraw

from detector import pose_estimation, synchrony_detection
from detector.synchrony_detection import SynchronyDetector


class Visualizer:
    def __init__(self):
        self.img = None
        self.poses = None
        self.relevant_poses = None
        self.synchrony = None
        self.synch_metric = None
        self.distance = None

    def setup(
        self, img, poses, relevant_poses, synchrony, synch_style, distance
    ):
        self.img = img
        self.poses = poses
        self.relevant_poses = relevant_poses
        self.synchrony = synchrony
        self.synch_metric = synch_style
        self.distance = distance

    def draw_bounding_boxes(self, track):
        for pose in self.poses:
            cv2.rectangle(
                self.img,
                (pose.bbox[0], pose.bbox[1]),
                (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]),
                (0, 255, 0),
            )
            if track:
                cv2.putText(
                    self.img,
                    "id: {}".format(pose.id),
                    (pose.bbox[0], pose.bbox[1] - 16),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 255),
                )

    def define_skeleton_color(self, pose, key, value):
        # condition 1
        bodypart_synch_score_available = value != -1
        # condition 2
        color_should_be_mirrored = (
            self.synch_metric in SynchronyDetector.synch_styles_mirrored
            and pose == self.relevant_poses[1]
        )

        if not bodypart_synch_score_available:
            color = [0, 0, 0]
        elif color_should_be_mirrored:
            mirror_key = synchrony_detection.get_mirror_key(key)
            color = [
                0,
                min(255, 2 * 255 * self.synchrony.get(mirror_key)),
                min(255, 2 * 255 * (1 - self.synchrony.get(mirror_key))),
            ]
        else:
            color = [
                0,
                min(255, 2 * 255 * value),
                min(255, 2 * 255 * (1 - value)),
            ]
        return color

    def skeleton_overlay(self):
        skeleton_keypoint_pairs = (
            pose_estimation.PoseEstimator.skeleton_keypoint_pairs
        )
        for pose in self.poses:
            colors = []
            if self.relevant_poses is None or pose not in self.relevant_poses:
                colors = [
                    [0, 0, 0] for _ in range(len(skeleton_keypoint_pairs))
                ]
            else:
                for key, value in self.synchrony.items():
                    color = self.define_skeleton_color(pose, key, value)
                    colors.append(color)
            pose.draw(self.img, colors)

    def text_overlay(self):
        if len(self.poses) < 2:
            overlay_text = f"Avg synch: {np.nan}; Dist: {np.nan} px"
            dash = self.overlay_dashboard(
                overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
        else:
            synch_vals_avail = [
                val for val in self.synchrony.values() if val != -1
            ]
            synch_mean = (
                mean(synch_vals_avail) if len(synch_vals_avail) != 0 else False
            )
            dist_temp = self.distance
            text_dist = np.nan if dist_temp == -1 else f"{dist_temp:.1f}px"
            text_synch = np.nan if synch_mean is False else f"{synch_mean:.1f}"
            overlay_text = f"Avg synch: {text_synch}; Dist: {text_dist}"
            dash = self.overlay_dashboard(
                overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )

        x_offset = y_offset = 10
        y_start = y_offset
        y_stop = y_offset + dash.shape[0]
        x_start = x_offset
        x_stop = x_offset + dash.shape[1]
        self.img[y_start:y_stop, x_start:x_stop] = dash

    def round_rectangle(self, size, radius, fill):
        """Draw a rounded rectangle"""
        width, height = size
        rectangle = Image.new("RGB", size, fill)
        corner = self.round_corner(radius, fill)
        rectangle.paste(corner, (0, 0))
        rectangle.paste(
            corner.rotate(90), (0, height - radius)
        )  # Rotate the corner and paste it
        rectangle.paste(corner.rotate(180), (width - radius, height - radius))
        rectangle.paste(corner.rotate(270), (width - radius, 0))
        return rectangle

    @staticmethod
    def round_corner(radius, fill):
        """Draw a round corner"""
        corner = Image.new("RGB", (radius, radius), (0, 0, 0, 0))
        draw = ImageDraw.Draw(corner)
        draw.pieslice((0, 0, radius * 2, radius * 2), 180, 270, fill=fill)
        return corner

    @staticmethod
    def draw_text(
        img,
        text,
        font=cv2.FONT_HERSHEY_PLAIN,
        pos=(0, 0),
        font_scale=3,
        font_thickness=2,
        text_color=(0, 255, 0),
        text_color_bg=(0, 0, 0),
    ):
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
        cv2.putText(
            img,
            text,
            (x, y + text_h + font_scale - 1),
            font,
            font_scale,
            text_color,
            font_thickness,
        )

        return text_size

    def overlay_dashboard(self, text, font, font_scale, font_thickness):
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        background = self.round_rectangle(
            (textsize[0], textsize[1] * 2), 10, "white"
        )
        dash = np.array(background)
        # Convert RGB to BGR
        dash = dash[:, :, ::-1].copy()
        # get coords based on boundary
        textX = (dash.shape[1] - textsize[0]) / 2
        textY = (dash.shape[0] + textsize[1]) / 2

        # add text centered on image
        cv2.putText(dash, text, (int(textX), int(textY)), font, 1, (0, 0, 0), 2)

        return dash
