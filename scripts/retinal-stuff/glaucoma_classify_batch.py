import argparse
import os.path
import time

import torch
import torch.nn.functional as tochfunc
import torchvision.transforms as tvtransf
from torch.ao.quantization import FakeQuantize, MinMaxObserver, PerChannelMinMaxObserver, MovingAverageMinMaxObserver
from torch.quantization import QConfig

from retina_classifier_networks import FcnskipNerveDefinitor2
from retinal_utils import get_bounding_box_fast, open_image, get_bounding_box_rectanglified, extract_objects_with_contours_np_cv2

