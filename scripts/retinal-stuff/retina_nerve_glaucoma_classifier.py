import os
import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import pandas as pd

# Import provided model and utilities
from retina_classifier_networks import HandmadeGlaucomaClassifier
from retinal_utils import open_image

# Constants matching definitor logic and network requirements
IMG_SHAPES = [576, 576, 3]
LOAD_MODE = "RGB"
DATA_LABELS_ORDERED = ['glaucoma', 'atrophy', 'valid_image']
MODEL_BASE = 64

def load_classifier(weights_pretrained_path, use_cuda=True):
    """
    Loads the HandmadeGlaucomaClassifier model using the specific state_dict
    structure found in training checkpoints.
    """
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

    # Load the full state dict
    state_dicts = torch.load(weights_pretrained_path, map_location=device)

    # Initialize model (Note: input_size is half of original 576 as per your logic)
    classifier = HandmadeGlaucomaClassifier(
        input_size=int(IMG_SHAPES[0] / 2),
        num_classes=len(DATA_LABELS_ORDERED),
        base=MODEL_BASE
    ).to(device)

    # Access the specific key used in your save utility
    if 'nerve_classifier' in state_dicts:
        classifier.load_state_dict(state_dicts['nerve_classifier'])
    else:
        # Fallback for raw state dicts
        classifier.load_state_dict(state_dicts)

    classifier.eval()
    return classifier, device


def run_classifier(image_path, loaded_classifier, premade_device):
    """
    Runs classification on a single cropped nerve image.
    Returns a dictionary of labels and their rounded probabilities.
    """
    # 1. Loading and Pre-processing
    pil_image_origin = open_image(image_path, load_mode=LOAD_MODE)

    # 2. Tensor conversion and scaling (Matching the 0.5 scale factor)
    transform_pipeline = T.Compose([
        T.ToTensor(),
        T.Resize(
            (int(IMG_SHAPES[0] / 2), int(IMG_SHAPES[1] / 2)),
            antialias=True,
            interpolation=InterpolationMode.BILINEAR
        )
    ])

    tensor_image = transform_pipeline(pil_image_origin).to(premade_device)

    # Normalize [0, 1] -> [-1, 1]
    tensor_image.mul_(2).sub_(1)

    # Ensure 3-channel (RGB) if single-channel input
    if tensor_image.size(0) == 1:
        tensor_image = torch.cat([tensor_image] * 3, dim=0)

    # 3. Inference
    tensor_image = tensor_image.unsqueeze(0)
    with torch.no_grad():
        output = loaded_classifier(tensor_image).squeeze(0)

    # 4. Result formatting
    # Convert to CPU numpy for DataFrame compatibility
    output_np = output.detach().cpu().numpy().reshape(1, -1)
    data = pd.DataFrame(output_np, columns=DATA_LABELS_ORDERED)
    data = data.round(5)

    # Return as a dictionary for easier orchestration in batch processing
    return data.to_dict(orient='records')[0]