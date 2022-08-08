# Entanglement Detector

## Prerequisites
Go to project folder /entanglement_detector and...
1. Install the requirements
~~~
pip install -r requirements.txt
~~~
2. [Download pose estimation checkpoint](https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth) from the project [OpenPose Lightweight](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)

3. Activate venv
~~~
source venv/bin/activate
~~~

## Instructions
To run the Entanglement Detector from your console, use the following below.
Default settings are:
- Use webcam as input
- Use synchronization style "2pax_90"
~~~
python -m detector.main --checkpoint-path <path_to>/checkpoint_iter_370000.pth
~~~

## Synchronization styles
You can choose from the synchronization styles listed below. Just use them as argument "synch-style" when calling the porgram from your command line.

### 2pax_90
Takes 2 most prominent people into account (ranked by confidence score of pose estimation). Lowest synchrony when body parts are angled at 90 degrees. High synchrony for parallel body parts, both at 0 or 180 degrees.

### 2pax_180
Takes 2 most prominent people into account (ranked by confidence score of pose estimation). Lowest synchrony when body parts are twisted at 180 degrees. High synchrony for parallel body parts at 0 degrees.

### 2pax_90_mirrored
Like "2 pax, 90 degrees" with the difference that mirrored body parts are compared. For example, the left upper arm of person A is compared to the right upper arm of person B to calculate synchronization.

### 2pax_180_mirrored
Like "2 pax, 180 degrees" with the difference that mirrored body parts are compared. For example, the left upper arm of person A is compared to the right upper arm of person B to calculate synchronization.

### allpax
Takes into account all persons detected by the pose estimation model. Synchronization is the normalized summed difference from the average directional vector over all persons (by body part).

### x, where x is an integer you choase
Takes the x most prominent people into account - or the total number of detected people if x > total number of people. Synchronization is the normalized summed difference from the average directional vector over all persons (by body part).
