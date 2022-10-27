import cv2
import pandas as pd


class OutputHandler:
    """
    Responsible for creating output tables and videos if the user chooses
    to save them.
    """

    def __init__(self, output_type: str, file_name: str, fps):
        if output_type not in ["video", "table"]:
            raise ValueError(
                "Invalid argument. " "Output_type must be 'video' or 'table'."
            )
        self.output_type = output_type
        self.file_name = file_name
        self.fps = fps
        self.table_writer = None
        self.video_writer = None
        self.setup_done = False

    def build_outputs(self, data):
        if self.output_type == "table":
            self.attach_to_output_table(data)
        if self.output_type == "video":
            self.attach_to_output_video(data)

    def attach_to_output_table(self, row: dict):
        if self.setup_done is False:  # first iteration
            self.table_writer = []
            self.setup_done = True
        self.table_writer.append(row)

    def attach_to_output_video(self, frame):
        if self.setup_done is False:  # first iteration
            frame_width = frame.shape[0]
            frame_height = frame.shape[1]
            fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
            self.video_writer = cv2.VideoWriter(
                self.file_name, fourcc, self.fps, (frame_height, frame_width)
            )
            self.setup_done = True
        self.video_writer.write(frame)

    def release_outputs(self):
        if self.output_type == "table":
            self.table_writer = pd.DataFrame(self.table_writer)
            self.table_writer.to_csv(self.file_name, index=False)
            self.setup_done = False
        elif self.output_type == "video":
            self.video_writer.release()
            # self.setup_done = False
