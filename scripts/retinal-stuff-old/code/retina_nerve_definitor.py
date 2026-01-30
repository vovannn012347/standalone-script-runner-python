import os
import torch
import torch.nn.functional as tochfunc
import torchvision.transforms as tvtransf

# Import provided model and utilities
from retina_networks import FCNSkipNerveDefinitorPrunnable
from retinal_utils import (
    open_image, get_bounding_box_fast, get_bounding_box_rectanglified,
    extract_objects_with_contours_np_cv2, magic_wand_mask_selection_faster,
    apply_clahe_lab, histogram_equalization_hsv_s
)

# Constants based on your provided config/test scripts
IMG_SHAPES = [576, 576, 3]
LOAD_MODE = "RGB"
MODEL_BASE = 64


def load_definitor(weights_pretrained_path, use_cuda=True):
    """
    Loads the FcnskipNerveDefinitor2 model using the specific state_dict
    structure found in the training/testing scripts.
    """
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

    # Load the full state dict
    state_dicts = torch.load(weights_pretrained_path, map_location=device)

    # Initialize model matching your training setup
    loaded_definitor = FCNSkipNerveDefinitorPrunnable(
        num_classes=1,
        # use_dropout=False,  # Eval mode typically disables dropout
        base=MODEL_BASE
    ).to(device)

    # Access the specific key used in your save_nerve_definitor utility
    if 'nerve_definitor' in state_dicts:
        loaded_definitor.load_state_dict(state_dicts['nerve_definitor'])
    else:
        # Fallback for raw state dicts
        loaded_definitor.load_state_dict(state_dicts)

    loaded_definitor.eval()
    return loaded_definitor, device


def run_definitor(image_path, output_path, loaded_definitor, premade_device):
    """
    Runs inference on a single image, processes the mask, and saves cropped results.
    """
    image_name = os.path.basename(image_path)
    image_name_no_ext, image_ext = os.path.splitext(image_name)

    # 1. Pre-processing pipeline
    pil_image_origin = open_image(image_path, load_mode=LOAD_MODE)

    # Image enhancement as requested in your snippet
    pil_image_processed = histogram_equalization_hsv_s(pil_image_origin)
    pil_image_processed = apply_clahe_lab(pil_image_processed)
    pil_image_processed = pil_image_processed.convert("L")

    # 2. Tensor conversion and scaling
    # Resize to base training size
    transform_pipeline = tvtransf.Compose([
        tvtransf.ToTensor(),
        tvtransf.Resize(IMG_SHAPES[:2], antialias=True)
    ])

    tensor_image = transform_pipeline(pil_image_processed).to(premade_device)
    pil_image_origin = pil_image_origin.resize(IMG_SHAPES[:2])
    _, h, w = tensor_image.shape

    # 3. Model Inference with 0.5 Interpolation (Match test_retina_nerve_definitor)
    input_tensor = tensor_image.unsqueeze(0)
    input_tensor = tochfunc.interpolate(
        input_tensor,
        scale_factor=0.5,
        mode='bilinear',
        align_corners=False
    )

    # Normalize [0, 1] -> [-1, 1]
    input_tensor.mul_(2).sub_(1)

    # Ensure 3-channel if required by model (as per your snippet logic)
    if input_tensor.size(1) == 1:
        input_tensor = torch.cat([input_tensor] * 3, dim=1)

    with torch.no_grad():
        output = loaded_definitor(input_tensor).squeeze(0)

    # 4. Mask Post-Processing
    # Hard thresholding and Magic Wand refinement
    output[output < 0.09] = 0
    output_wand_selected = magic_wand_mask_selection_faster(
        output,
        upper_multiplier=0.15,
        lower_multipleir=0.3
    ).to(torch.float32)

    # 5. Object Extraction and Cropping
    _, h_bb, w_bb = output_wand_selected.shape
    split_tensors = extract_objects_with_contours_np_cv2(output_wand_selected)

    out_filenames = []
    expand_constant = 0.2

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for split_idx, tensor in enumerate(split_tensors):
        img_bbox = get_bounding_box_fast(tensor)  # [left, top, right, bottom]

        # Calculate expanded bounding box
        bb_w = img_bbox[2] - img_bbox[0]
        bb_h = img_bbox[3] - img_bbox[1]

        # Normalize and clip
        x1 = max((img_bbox[0] - bb_w * expand_constant) / w_bb, 0)
        y1 = max((img_bbox[1] - bb_h * expand_constant) / h_bb, 0)
        x2 = min((img_bbox[2] + bb_w * expand_constant) / w_bb, 1.0)
        y2 = min((img_bbox[3] + bb_h * expand_constant) / h_bb, 1.0)

        # Scale back to original resolution
        final_bbox = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
        final_bbox = get_bounding_box_rectanglified(final_bbox, h, w)

        # Valid crop check
        if (final_bbox[2] - final_bbox[0]) > 1 and (final_bbox[3] - final_bbox[1]) > 1:
            pil_image_cropped = pil_image_origin.crop(final_bbox)
            out_filename = f"{image_name_no_ext}_cropped_{split_idx}{image_ext}"

            save_path = os.path.join(output_path, out_filename)
            pil_image_cropped.save(save_path)
            out_filenames.append(save_path)

    return out_filenames
