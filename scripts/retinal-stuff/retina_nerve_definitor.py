import os.path
import time

import torch
import torch.nn.functional as tochfunc
import torchvision.transforms as tvtransf

from retina_classifier_networks import FcnskipNerveDefinitor2
from retinal_utils import open_image, get_bounding_box_fast,  get_bounding_box_rectanglified, \
    extract_objects_with_contours_np_cv2, magic_wand_mask_selection_faster, apply_clahe_lab, \
    histogram_equalization_hsv_s

img_shapes = [576, 576, 3]
load_mode = "RGB"
model_base = 64


def split_by_position(string, position):
    return [string[:position], string[position:]]


def load_definitor(weights_pretrained_path):
    use_cuda_if_available = False
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_available else 'cpu')

    convolution_state_dict = torch.load(weights_pretrained_path, map_location=device)

    loaded_definitor = FcnskipNerveDefinitor2(
        num_classes=1,
        use_dropout=False,
        base=model_base).to(device)

    loaded_definitor.load_state_dict(convolution_state_dict['nerve_definitor'])
    loaded_definitor.eval()
    return loaded_definitor


def run_definitor(image_path, output_path, loaded_definitor):  # , weights_pretrained_path = None):

    # loaded_definitor = load_definitor(weights_pretrained_path)

    image_name = os.path.basename(image_path)  # file name
    image_name, image_ext = os.path.splitext(image_name)  # name and extension

    pil_image_origin = open_image(image_path, load_mode=load_mode)  # Image.open(args.image).convert('RGB')  # 3 channel

    # apply histogram equalization
    pil_image_processed = histogram_equalization_hsv_s(pil_image_origin)
    pil_image_processed = apply_clahe_lab(pil_image_processed)
    pil_image_processed = pil_image_processed.convert("L")

    tensor_image = tvtransf.ToTensor()(pil_image_processed)
    tensor_image = tvtransf.Resize(img_shapes[:2], antialias=True)(tensor_image)
    pil_image_origin = pil_image_origin.resize(img_shapes[:2])
    channels, h, w = tensor_image.shape

    # detection is done on scaled down image
    tensor_image = tochfunc.interpolate(tensor_image.unsqueeze(0),
                                        scale_factor=0.5,
                                        mode='bilinear',
                                        align_corners=False).squeeze(0)

    tensor_image.mul_(2).sub_(1)  # [0, 1] -> [-1, 1]
    if tensor_image.size(0) == 1: # single-channel -> RGB. redundant but just in case
        tensor_image = torch.cat([tensor_image] * 3, dim=0)

    tensor_image = tensor_image.unsqueeze(0)
    output = loaded_definitor(tensor_image).squeeze(0)

    to_pil_transform = tvtransf.ToPILImage(mode='L')

    output[output < 0.09] = 0
    # output = torch.clamp(output, 0.09, 1)
    output_wand_selected = (magic_wand_mask_selection_faster(output, upper_multiplier=0.15, lower_multipleir=0.3)
                              .to(torch.float32))

    channels_bb, h_bb, w_bb = output_wand_selected.shape
    split_tensors = extract_objects_with_contours_np_cv2(output_wand_selected)

    out_filenames = []

    for split_idx, tensor in enumerate(split_tensors):

        img_bbox = get_bounding_box_fast(tensor)  # left, top, right, bottom

        expand_constant = 0.2
        bb_w = img_bbox[2] - img_bbox[0]
        bb_h = img_bbox[3] - img_bbox[1]
        img_bbox2 = (img_bbox[0] - bb_w * expand_constant) / w_bb, (img_bbox[1] - bb_h * expand_constant) / h_bb, (
                    img_bbox[2] + bb_w * expand_constant) / w_bb, (img_bbox[3] + bb_h * expand_constant) / h_bb
        img_bbox3 = max(img_bbox2[0], 0), max(img_bbox2[1], 0), min(img_bbox2[2], 1.0), min(img_bbox2[3], 1.0),
        img_bbox4 = int(img_bbox3[0] * h), int(img_bbox3[1] * w), int(img_bbox3[2] * h), int(img_bbox3[3] * w)

        img_bbox4 = get_bounding_box_rectanglified(img_bbox4, h, w)

        if (img_bbox4[2] - img_bbox4[0]) > 1 and (img_bbox4[3] - img_bbox4[1]) > 1:
            pil_image_cropped = pil_image_origin.crop(img_bbox4)
            # pil_image_cropped.save(args.out_cropped)
            out_filename = f"{image_name}_cropped_{split_idx}{image_ext}"
            image_cropped_path_out = os.path.join(output_path, out_filename)
            pil_image_cropped.save(image_cropped_path_out)
            out_filenames.append(out_filename)

    return out_filenames


