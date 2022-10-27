# Entanglement Detector

### System requirements

-   Availability of Python version 3.8 or higher.

-   If analysis should be run on GPU: Compatibility of GPU with NVIDIA
    CUDA.

-   If system should run analyses for live input streams: Availability
    of a connected camera device.

### Installation and Initial Setup

- Source files need to be added to local device.
  ~~~
  git clone git@github.com:strgfjojo/entanglement-detector.git
  ~~~

- Dependencies need to be installed.
  ~~~
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
  ~~~

- Pose estimation checkpoint [file](https://bit.ly/3f8sfNw) for pre-trained
- [Lightweight OpenPose](https://bit.ly/3Dfwi2u) model needs to be downloaded.

### Operating Command

To run the system, arguments have to be provided for at least the
`--video` and `--checkpoint-path` parameters. For example, the following
command line prompt runs synchronization analysis for a connected camera
input:

    python3 -m main
    --checkpoint-path <path_to>/checkpoint_iter_370000.pth
    --video 0

In the following, additional input parameters are described.

### Input Parameters

`--video`[Required. No default]\
Specifies the path to the input video. Can be stated as the absolute
path or the path from the repository root. If a connected recording
device should be used, its index can be specified. Index is `0` if only
a single camera is connected to the computer. An argument for this
parameter is required.

`--checkpoint-path` [Required. No default]\
Specifies path to the checkpoint file. The weights file for the pose
estimation module is not provided in the GitHub repository of the
system. It needs to be downloaded separately (
[file here](https://bit.ly/3f8sfNw)) and is provided by
[Lightweight OpenPose](https://bit.ly/3Dfwi2u).
An argument for this parameter is required.

`--synch-metric`[Default: pss]\
Specifies the synchronization metric to be applied.
The following arguments can be stated:

- **`pss`**   [default] implements perpendicular same-side synchronization.
Comparing the same body sides, a body part is considered fully ill-synchronized
if vectors are perpendicular to each other. Perfect synchronization results
from parallel vectors, regardless of whether they point in the same or opposite
direction.

- **`pos`**   implements perpendicular opposite-side synchronization.
Comparing opposing body sides, a body part is considered fully ill-synchronized
if vectors are perpendicular to each other. Perfect synchronization results
from parallel vectors, regardless of whether they point in the same or
opposite direction.

- **`lss`**   implements linear same-side synchronization.
Comparing the same body sides, a body part is considered perfectly
synchronized if vectors point into the same direction. Synchronization
decreases with increasing angle between the vectors. Synchronization scores
reach zero if vectors point into opposing directions.

- **`los`**   implements linear opposite-side synchronization. Comparing
opposing body sides, a body part is considered perfectly synchronized if
vectors point into the same direction. Synchronization decreases with
increasing angle between the vectors. Synchronization scores reach zero if
vectors point into opposing directions.

`--group-size`[Default: 2]\
Specifies the number of people to be included in the synchronization and
distance calculations. If a specific number should be quantified, the
argument must be a natural number greater or equal than 2. If the argument
exceeds the number of people visible in the input, synchronization and distance
are calculated for all input poses. By default, the argument is `2`. Poses
are chosen according to their confidence values. Alternatively, **`all`**
instructs the system to include all poses in the synchronization and
distance calculations.

`--show-livestream`[Default: True]\
Specifies if a real-time visual output stream should be displayed that
illustrates the estimated synchronization. If set to `True`, the live
stream is displayed. If set to `False`, it is not displayed. If no
argument is specified, the livestream will be displayed by default.

`--save-livestream`[Default: False]\
Specifies if the visual output stream that illustrates the estimated
synchronization should be stored. If set to `True`, the live stream is
stored in the project directory. If set to `False`, the frames of the
live stream will be discarded after display. Per default,
`--save-livestream` is set `False`.

`--save-output-table`[Default: False]\
Specifies if a tabular output file should be created and stored. The
file stores body part synchronizations and body distance for each frame
analyzed. If set to `True`, a \".csv\" file is stored in the project
directory. The default argument is `False`.

`--save-camera-input`[Default: False]\
Specifies if a video should be created and stored that duplicates the
input video without alterations. This parameter can be relevant if the
system uses a connected recording device as an input source, whose
captured frames will otherwise be discarded.

`--cpu`[Default: True]\
Specifies if CPU or GPU should be used for processing. If set to `True`,
the CPU will be used. CPU is used also by default. If set to `False`,
processing will be executed on the GPU. This option only supports GPU
usage if the GPU is compatible with
[NVIDIA CUDA](<https://docs.nvidia.com/cuda/>).

`--frame-skip`[Default: 1]\
Specifies that only every `--frame-skip`-th frame shall be analyzed.
Set to `1` by default, which means that every frame gets analyzed.
Can be used to speed up analysis.


### Outputs

`output_video.avi`\
Output video that visualizes the detected synchronization on the input
video. File is created and stored in the project directory if
`--save-output-table` is set `True`.

`input_video.avi`\
Video that duplicates the input video without alterations. File is
created and stored in the project directory if is set `True`.

`output_table.csv`\
File that stores time series of body part synchronizations and body
distance in tabular form. File is created and stored in the project
directory if `--save-output-table` is set `True`.
