import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

# Import provided model and utilities
from retina_networks import HandmadeGlaucomaClassifier
from retinal_utils import open_image

# Constants matching established architecture
DATA_LABELS_ORDERED = ['glaucoma', 'atrophy', 'valid_image']
IMG_SHAPES = [128, 128, 3]
MODEL_BASE = 64


def load_classifier(weights_pretrained_path, use_cuda=True):
    """Loads model using standard PyTorch without numpy/pandas."""
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    state_dicts = torch.load(weights_pretrained_path, map_location=device)

    classifier = HandmadeGlaucomaClassifier(
        input_size=IMG_SHAPES[0],
        num_classes=len(DATA_LABELS_ORDERED),
        base=MODEL_BASE
    ).to(device)

    if 'nerve_classifier' in state_dicts:
        classifier.load_state_dict(state_dicts['nerve_classifier'])
    else:
        classifier.load_state_dict(state_dicts)

    classifier.eval()
    return classifier, device


def run_classifier(image_path, loaded_classifier, premade_device):
    """Inference with [-1, 1] normalization and no numpy."""
    pil_image = open_image(image_path)

    input_size = IMG_SHAPES[0]
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((input_size, input_size), antialias=True, interpolation=InterpolationMode.BILINEAR)
    ])

    tensor_image = transform(pil_image).to(premade_device)
    tensor_image = tensor_image.mul(2.0).sub(1.0)  # Scaling [0,1] -> [-1,1]

    if tensor_image.size(0) == 1:
        tensor_image = torch.cat([tensor_image] * 3, dim=0)

    with torch.no_grad():
        output = loaded_classifier(tensor_image.unsqueeze(0)).squeeze(0)

    # Convert to standard Python list and zip with labels
    output_list = output.detach().cpu().tolist()
    results = {label: round(val, 5) for label, val in zip(DATA_LABELS_ORDERED, output_list)}

    return results
