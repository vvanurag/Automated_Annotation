# Automated_Annotation
Automated Annotation using Grounded SAM and the likes

## Discovery line
**Autodistil**: Autodistill uses big, slower foundation models to train small, faster supervised models
            You can use Autodistill on your own hardware, or use the Roboflow hosted version of Autodistill to label images in the cloud.

**Roboflow Auto Label (beta)**: Many of the top models used in production today like YOLOv9, Ultralytics YOLOv8, and YOLO-NAS require
                                 fine-tuning on hundreds or thousands of labeled images to achieve the accuracy necessary for production systems.
                                 Auto labels use foundation models like Grpinded Dino, Grounded SAM, which are slow to use in production but are 
                                 zero-shot models(Definition)


## Definitions
Zero shot models
Grounded models

## Grounded SAM2 for automated annotation: segmentation and bounding box.

## Install Grounded SAM 2
* ### Fresh start
* conda create -n grounded-sam2 python=3.10 -y
* conda activate grounded-sam2
* conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
* pip3 install autodistill-grounded-sam-2
---
### After that, start writing code.
* from autodistill_grounded_sam_2 import GroundedSAM2

*** Code Block ***

## Test code
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import numpy as np
import cv2
import os
import random
import supervision as sv
from autodistill_grounded_sam_2.grounded_sam_2 import GroundedSAM2

#### Define an ontology to map class names to the Grounded SAM 2 prompt
#### The ontology dictionary has the format {caption: class}
#### where 'caption' is the prompt sent to the base model, and 'class' is the label saved
ontology = CaptionOntology(
    {
        "ground": "ground"
    }
)

#### Initialize the model without the 'model' argument
base_mdl = GroundedSAM2(
    ontology=ontology
)

#### Set additional parameters directly (if needed)
base_mdl.grounding_dino_box_threshold = 0.50
*** Code block close ***



