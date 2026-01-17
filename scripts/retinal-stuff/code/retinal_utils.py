from PIL import Image
from torch import Tensor
import torch
import torch.nn.functional as tnnfunc

import cv2
import numpy as np


def open_image(image_path, load_mode='RGB'):
    pil_image = pil_loader(image_path, load_mode)
    img_bbox = get_image_bbox(pil_image)

    pil_image = pil_image.crop(img_bbox)
    return pil_image


def pil_loader(path, mode='RGB'):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(mode)


def get_image_bbox(image_pil):
    # convert image to grayscale,
    # get the bounding box of non-black regions
    gray_image = image_pil.convert("L")

    width, height = gray_image.size
    top, left, bottom, right = height, width, 0, 0
    threshold = 10

    # Iterate over each pixel
    for y in range(height):
        for x in range(width):
            # Get pixel value
            pixel = gray_image.getpixel((x, y))

            # Check if pixel is almost black
            if pixel > threshold:
                # Update bounding box coordinates
                top = min(top, y)
                left = min(left, x)
                bottom = max(bottom, y)
                right = max(right, x)

    return left, top, right, bottom


def get_bounding_box_fast(image_tensor: Tensor):
    # Check for non-zero values along each axis to find the bbox
    img = image_tensor.squeeze(0)
    rows = torch.any(img > 0, dim=1)  # True for rows with non-zero pixels
    cols = torch.any(img > 0, dim=0)  # True for columns with non-zero pixels

    # Get min and max of non-zero rows and columns
    min_y, max_y = (0, 1)
    min_x, max_x = (0, 1)

    search_y = torch.where(rows)
    search_x = torch.where(cols)

    if search_y[0].size(0) > 0:
        min_y, max_y = torch.where(rows)[0][[0, -1]]
        min_y, max_y = (min_y.item(), max_y.item())

    if search_x[0].size(0) > 0:
        min_x, max_x = torch.where(cols)[0][[0, -1]]
        min_x, max_x = (min_x.item(), max_x.item())

    # left, top, right, bottom
    bbox = (min_x, min_y, max_x, max_y)
    return bbox


def get_bounding_box_rectanglified(bbox, img_width, img_height, rect_min_w=128, rect_min_h=128):
    # bbbox order: left, top, right, bottom

    x_min = bbox[0]
    y_min = bbox[1]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    x_center = x_min + width / 2
    y_center = y_min + height / 2

    if width < rect_min_w:
        width = rect_min_w

    if height < rect_min_h:
        height = rect_min_h

    if height > img_height:
        height = img_height

    if width > img_width:
        width = img_width

    side_length = max(width, height)
    x_new_min = x_center - side_length / 2
    y_new_min = y_center - side_length / 2

    x_new_min = max(0, min(x_new_min, img_width - side_length))
    y_new_min = max(0, min(y_new_min, img_height - side_length))

    return (int(x_new_min),
            int(y_new_min),
            min(int(x_new_min + side_length), img_width),
            min(int(y_new_min + side_length), img_height))


def extract_objects_with_contours_np_cv2(tensor):

    if tensor.ndim == 3:  # If shape is (1, H, W), squeeze to (H, W)
        tensor = tensor.squeeze(0)
    binary_mask = tensor.cpu().numpy().astype(np.uint8)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tensors = []

    for idx, contour in enumerate(contours):

        object_mask = np.zeros_like(binary_mask)
        cv2.drawContours(object_mask, [contour], -1, 255, thickness=cv2.FILLED)

        tensor = torch.from_numpy(object_mask).unsqueeze(0)
        tensors.append(tensor)

    return tensors


def magic_wand_mask_selection_faster(image_tensor, upper_multiplier=0.4, lower_multipleir=0.25):
    # , debug_dir):
    """ selects retilnal blood vessels from mask via magic wand

        Args:
            image_tensor (Tensor): 1 channel tensor that contains greyscale values for mask.
            upper_multiplier (float): tolerance multiplier that decides how far the first wand selection goes
            lower_multipleir (float): tolerance multiplier that decides how far the second wand selection goes

        Returns:
            Tensor: Tensor with boolean values that denote selected pixels.
        """
    if image_tensor.dim() != 3 or image_tensor.size(0) != 1:  # RGB
        raise Exception("invalid image_tensor dimensions")

    # part 1: get above zero pixel values
    flat_image = image_tensor.flatten()

    bin_count = 256
    min_pixel = 0.0  # torch.min(flat_image).item()
    max_pixel = 1.0  # torch.max(flat_image).item()
    histogram = torch.histc(flat_image.float(), bins=bin_count, min=0.0, max=1.0)

    # part 2: get starting tolerance and starting pixel value
    bin_width = (max_pixel - min_pixel) / bin_count
    non_zero_indices = torch.nonzero(histogram, as_tuple=False)

    first_tolerance = upper_multiplier

    first_bound_bin_index = int(non_zero_indices[-1].item() * (1 - first_tolerance))
    first_bound = first_bound_bin_index * bin_width

    if first_bound <= 1e-8:
        return torch.zeros_like(image_tensor)

    # part 3: make starting global selection
    mask = image_tensor > first_bound

    # part 4: replace selected pixel values with the lowest value from selected pixels
    image_tensor_wand = torch.clone(image_tensor)
    image_tensor_wand[mask] = first_bound

    # part 5: get second tolerance value
    # from the lowest color value in part 2 to total lowest value that is above 0 color
    lower_bound_bin_index = int(first_bound_bin_index * lower_multipleir)
    if lower_bound_bin_index < 3:
        lower_bound_bin_index = 3
        if lower_bound_bin_index >= first_bound_bin_index:
            lower_bound_bin_index = first_bound_bin_index - 1

    if lower_bound_bin_index < 0:
        lower_bound_bin_index = 0

    lower_bound = lower_bound_bin_index * bin_width

    diff_map = ((image_tensor_wand - lower_bound) >= 0).squeeze()

    kernel = torch.tensor([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=torch.float32,
                          device=image_tensor.device).unsqueeze(0).unsqueeze(0)

    max_iters = int((mask.size(1) * mask.size(2)) / 4)

    # mask = mask.squeeze(0)

    for _ in range(max_iters):
        dilated_mask = (tnnfunc.conv2d(mask.float().unsqueeze(0), kernel, padding=1)
                        .squeeze().bool())

        # Mask update: keep pixels within threshold and add to current mask
        new_mask = dilated_mask & diff_map

        # Stop if no new pixels are added
        if torch.equal(new_mask, mask):
            break

        # Update mask with new selection
        mask = new_mask

    return mask.unsqueeze(0)


def apply_clahe_lab(image, clip_limit=2.0, tile_grid_size=(16, 16)):
    # Convert PIL image to a NumPy array
    image_np = np.array(image)

    # Convert RGB to LAB color space
    lab_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)

    # Split into L, A, and B channels
    l, a, b = cv2.split(lab_image)

    # Apply CLAHE to the L (luminance) channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    res = clahe.apply(l)

    # Merge the channels back and convert to RGB
    lab_image = cv2.merge((res, a, b))
    image_clahe_rgb = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

    # Convert back to PIL Image
    return Image.fromarray(image_clahe_rgb)


def histogram_equalization_hsv_s(image):
    image_np = np.array(image)

    hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    s_eq = cv2.equalizeHist(s)
    hsv_eq = cv2.merge((h, s_eq, v))

    image_equalized = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

    return Image.fromarray(image_equalized)


def histogram_equalization_hsv_v(image):
    image_np = np.array(image)

    hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge((h, s, v_eq))

    image_equalized = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

    return Image.fromarray(image_equalized)