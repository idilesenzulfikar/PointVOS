import cv2
import numpy as np

pascal_colormap = [
    0, 0, 0,
    0.5020, 0, 0,
    0, 0.5020, 0,
    0.5020, 0.5020, 0,
    0, 0, 0.5020,
    0.5020, 0, 0.5020,
    0, 0.5020, 0.5020,
    0.5020, 0.5020, 0.5020,
    0.2510, 0, 0,
    0.7529, 0, 0,
    0.2510, 0.5020, 0,
    0.7529, 0.5020, 0,
    0.2510, 0, 0.5020,
    0.7529, 0, 0.5020,
    0.2510, 0.5020, 0.5020,
    0.7529, 0.5020, 0.5020,
    0, 0.2510, 0,
    0.5020, 0.2510, 0,
    0, 0.7529, 0,
    0.5020, 0.7529, 0,
    0, 0.2510, 0.5020,
    0.5020, 0.2510, 0.5020,
    0, 0.7529, 0.5020,
    0.5020, 0.7529, 0.5020,
    0.2510, 0.2510, 0,
    0.7529, 0.2510, 0,
    0.2510, 0.7529, 0,
    0.7529, 0.7529, 0,
    0.2510, 0.2510, 0.5020,
    0.7529, 0.2510, 0.5020,
    0.2510, 0.7529, 0.5020,
    0.7529, 0.7529, 0.5020,
    0, 0, 0.2510,
    0.5020, 0, 0.2510,
    0, 0.5020, 0.2510,
    0.5020, 0.5020, 0.2510,
    0, 0, 0.7529,
    0.5020, 0, 0.7529,
    0, 0.5020, 0.7529,
    0.5020, 0.5020, 0.7529,
    0.2510, 0, 0.2510,
    0.7529, 0, 0.2510,
    0.2510, 0.5020, 0.2510,
    0.7529, 0.5020, 0.2510,
    0.2510, 0, 0.7529,
    0.7529, 0, 0.7529,
    0.2510, 0.5020, 0.7529,
    0.7529, 0.5020, 0.7529,
    0, 0.2510, 0.2510,
    0.5020, 0.2510, 0.2510,
    0, 0.7529, 0.2510,
    0.5020, 0.7529, 0.2510,
    0, 0.2510, 0.7529,
    0.5020, 0.2510, 0.7529,
    0, 0.7529, 0.7529,
    0.5020, 0.7529, 0.7529,
    0.2510, 0.2510, 0.2510,
    0.7529, 0.2510, 0.2510,
    0.2510, 0.7529, 0.2510,
    0.7529, 0.7529, 0.2510,
    0.2510, 0.2510, 0.7529,
    0.7529, 0.2510, 0.7529,
    0.2510, 0.7529, 0.7529,
    0.7529, 0.7529, 0.7529,
    0.1255, 0, 0,
    0.6275, 0, 0,
    0.1255, 0.5020, 0,
    0.6275, 0.5020, 0,
    0.1255, 0, 0.5020,
    0.6275, 0, 0.5020,
    0.1255, 0.5020, 0.5020,
    0.6275, 0.5020, 0.5020,
    0.3765, 0, 0,
    0.8784, 0, 0,
    0.3765, 0.5020, 0,
    0.8784, 0.5020, 0,
    0.3765, 0, 0.5020,
    0.8784, 0, 0.5020,
    0.3765, 0.5020, 0.5020,
    0.8784, 0.5020, 0.5020,
    0.1255, 0.2510, 0,
    0.6275, 0.2510, 0,
    0.1255, 0.7529, 0,
    0.6275, 0.7529, 0,
    0.1255, 0.2510, 0.5020,
    0.6275, 0.2510, 0.5020,
    0.1255, 0.7529, 0.5020,
    0.6275, 0.7529, 0.5020,
    0.3765, 0.2510, 0,
    0.8784, 0.2510, 0,
    0.3765, 0.7529, 0,
    0.8784, 0.7529, 0,
    0.3765, 0.2510, 0.5020,
    0.8784, 0.2510, 0.5020,
    0.3765, 0.7529, 0.5020,
    0.8784, 0.7529, 0.5020,
    0.1255, 0, 0.2510,
    0.6275, 0, 0.2510,
    0.1255, 0.5020, 0.2510,
    0.6275, 0.5020, 0.2510,
    0.1255, 0, 0.7529,
    0.6275, 0, 0.7529,
    0.1255, 0.5020, 0.7529,
    0.6275, 0.5020, 0.7529,
    0.3765, 0, 0.2510,
    0.8784, 0, 0.2510,
    0.3765, 0.5020, 0.2510,
    0.8784, 0.5020, 0.2510,
    0.3765, 0, 0.7529,
    0.8784, 0, 0.7529,
    0.3765, 0.5020, 0.7529,
    0.8784, 0.5020, 0.7529,
    0.1255, 0.2510, 0.2510,
    0.6275, 0.2510, 0.2510,
    0.1255, 0.7529, 0.2510,
    0.6275, 0.7529, 0.2510,
    0.1255, 0.2510, 0.7529,
    0.6275, 0.2510, 0.7529,
    0.1255, 0.7529, 0.7529,
    0.6275, 0.7529, 0.7529,
    0.3765, 0.2510, 0.2510,
    0.8784, 0.2510, 0.2510,
    0.3765, 0.7529, 0.2510,
    0.8784, 0.7529, 0.2510,
    0.3765, 0.2510, 0.7529,
    0.8784, 0.2510, 0.7529,
    0.3765, 0.7529, 0.7529,
    0.8784, 0.7529, 0.7529,
    0, 0.1255, 0,
    0.5020, 0.1255, 0,
    0, 0.6275, 0,
    0.5020, 0.6275, 0,
    0, 0.1255, 0.5020,
    0.5020, 0.1255, 0.5020,
    0, 0.6275, 0.5020,
    0.5020, 0.6275, 0.5020,
    0.2510, 0.1255, 0,
    0.7529, 0.1255, 0,
    0.2510, 0.6275, 0,
    0.7529, 0.6275, 0,
    0.2510, 0.1255, 0.5020,
    0.7529, 0.1255, 0.5020,
    0.2510, 0.6275, 0.5020,
    0.7529, 0.6275, 0.5020,
    0, 0.3765, 0,
    0.5020, 0.3765, 0,
    0, 0.8784, 0,
    0.5020, 0.8784, 0,
    0, 0.3765, 0.5020,
    0.5020, 0.3765, 0.5020,
    0, 0.8784, 0.5020,
    0.5020, 0.8784, 0.5020,
    0.2510, 0.3765, 0,
    0.7529, 0.3765, 0,
    0.2510, 0.8784, 0,
    0.7529, 0.8784, 0,
    0.2510, 0.3765, 0.5020,
    0.7529, 0.3765, 0.5020,
    0.2510, 0.8784, 0.5020,
    0.7529, 0.8784, 0.5020,
    0, 0.1255, 0.2510,
    0.5020, 0.1255, 0.2510,
    0, 0.6275, 0.2510,
    0.5020, 0.6275, 0.2510,
    0, 0.1255, 0.7529,
    0.5020, 0.1255, 0.7529,
    0, 0.6275, 0.7529,
    0.5020, 0.6275, 0.7529,
    0.2510, 0.1255, 0.2510,
    0.7529, 0.1255, 0.2510,
    0.2510, 0.6275, 0.2510,
    0.7529, 0.6275, 0.2510,
    0.2510, 0.1255, 0.7529,
    0.7529, 0.1255, 0.7529,
    0.2510, 0.6275, 0.7529,
    0.7529, 0.6275, 0.7529,
    0, 0.3765, 0.2510,
    0.5020, 0.3765, 0.2510,
    0, 0.8784, 0.2510,
    0.5020, 0.8784, 0.2510,
    0, 0.3765, 0.7529,
    0.5020, 0.3765, 0.7529,
    0, 0.8784, 0.7529,
    0.5020, 0.8784, 0.7529,
    0.2510, 0.3765, 0.2510,
    0.7529, 0.3765, 0.2510,
    0.2510, 0.8784, 0.2510,
    0.7529, 0.8784, 0.2510,
    0.2510, 0.3765, 0.7529,
    0.7529, 0.3765, 0.7529,
    0.2510, 0.8784, 0.7529,
    0.7529, 0.8784, 0.7529,
    0.1255, 0.1255, 0,
    0.6275, 0.1255, 0,
    0.1255, 0.6275, 0,
    0.6275, 0.6275, 0,
    0.1255, 0.1255, 0.5020,
    0.6275, 0.1255, 0.5020,
    0.1255, 0.6275, 0.5020,
    0.6275, 0.6275, 0.5020,
    0.3765, 0.1255, 0,
    0.8784, 0.1255, 0,
    0.3765, 0.6275, 0,
    0.8784, 0.6275, 0,
    0.3765, 0.1255, 0.5020,
    0.8784, 0.1255, 0.5020,
    0.3765, 0.6275, 0.5020,
    0.8784, 0.6275, 0.5020,
    0.1255, 0.3765, 0,
    0.6275, 0.3765, 0,
    0.1255, 0.8784, 0,
    0.6275, 0.8784, 0,
    0.1255, 0.3765, 0.5020,
    0.6275, 0.3765, 0.5020,
    0.1255, 0.8784, 0.5020,
    0.6275, 0.8784, 0.5020,
    0.3765, 0.3765, 0,
    0.8784, 0.3765, 0,
    0.3765, 0.8784, 0,
    0.8784, 0.8784, 0,
    0.3765, 0.3765, 0.5020,
    0.8784, 0.3765, 0.5020,
    0.3765, 0.8784, 0.5020,
    0.8784, 0.8784, 0.5020,
    0.1255, 0.1255, 0.2510,
    0.6275, 0.1255, 0.2510,
    0.1255, 0.6275, 0.2510,
    0.6275, 0.6275, 0.2510,
    0.1255, 0.1255, 0.7529,
    0.6275, 0.1255, 0.7529,
    0.1255, 0.6275, 0.7529,
    0.6275, 0.6275, 0.7529,
    0.3765, 0.1255, 0.2510,
    0.8784, 0.1255, 0.2510,
    0.3765, 0.6275, 0.2510,
    0.8784, 0.6275, 0.2510,
    0.3765, 0.1255, 0.7529,
    0.8784, 0.1255, 0.7529,
    0.3765, 0.6275, 0.7529,
    0.8784, 0.6275, 0.7529,
    0.1255, 0.3765, 0.2510,
    0.6275, 0.3765, 0.2510,
    0.1255, 0.8784, 0.2510,
    0.6275, 0.8784, 0.2510,
    0.1255, 0.3765, 0.7529,
    0.6275, 0.3765, 0.7529,
    0.1255, 0.8784, 0.7529,
    0.6275, 0.8784, 0.7529,
    0.3765, 0.3765, 0.2510,
    0.8784, 0.3765, 0.2510,
    0.3765, 0.8784, 0.2510,
    0.8784, 0.8784, 0.2510,
    0.3765, 0.3765, 0.7529,
    0.8784, 0.3765, 0.7529,
    0.3765, 0.8784, 0.7529,
    0.8784, 0.8784, 0.7529]

detectron_colormap = [
    0.000, 0.447, 0.741,
    0.850, 0.325, 0.098,
    0.929, 0.694, 0.125,
    0.494, 0.184, 0.556,
    0.466, 0.674, 0.188,
    0.301, 0.745, 0.933,
    0.635, 0.078, 0.184,
    0.300, 0.300, 0.300,
    0.600, 0.600, 0.600,
    1.000, 0.000, 0.000,
    1.000, 0.500, 0.000,
    0.749, 0.749, 0.000,
    0.000, 1.000, 0.000,
    0.000, 0.000, 1.000,
    0.667, 0.000, 1.000,
    0.333, 0.333, 0.000,
    0.333, 0.667, 0.000,
    0.333, 1.000, 0.000,
    0.667, 0.333, 0.000,
    0.667, 0.667, 0.000,
    0.667, 1.000, 0.000,
    1.000, 0.333, 0.000,
    1.000, 0.667, 0.000,
    1.000, 1.000, 0.000,
    0.000, 0.333, 0.500,
    0.000, 0.667, 0.500,
    0.000, 1.000, 0.500,
    0.333, 0.000, 0.500,
    0.333, 0.333, 0.500,
    0.333, 0.667, 0.500,
    0.333, 1.000, 0.500,
    0.667, 0.000, 0.500,
    0.667, 0.333, 0.500,
    0.667, 0.667, 0.500,
    0.667, 1.000, 0.500,
    1.000, 0.000, 0.500,
    1.000, 0.333, 0.500,
    1.000, 0.667, 0.500,
    1.000, 1.000, 0.500,
    0.000, 0.333, 1.000,
    0.000, 0.667, 1.000,
    0.000, 1.000, 1.000,
    0.333, 0.000, 1.000,
    0.333, 0.333, 1.000,
    0.333, 0.667, 1.000,
    0.333, 1.000, 1.000,
    0.667, 0.000, 1.000,
    0.667, 0.333, 1.000,
    0.667, 0.667, 1.000,
    0.667, 1.000, 1.000,
    1.000, 0.000, 1.000,
    1.000, 0.333, 1.000,
    1.000, 0.667, 1.000,
    0.167, 0.000, 0.000,
    0.333, 0.000, 0.000,
    0.500, 0.000, 0.000,
    0.667, 0.000, 0.000,
    0.833, 0.000, 0.000,
    1.000, 0.000, 0.000,
    0.000, 0.167, 0.000,
    0.000, 0.333, 0.000,
    0.000, 0.500, 0.000,
    0.000, 0.667, 0.000,
    0.000, 0.833, 0.000,
    0.000, 1.000, 0.000,
    0.000, 0.000, 0.167,
    0.000, 0.000, 0.333,
    0.000, 0.000, 0.500,
    0.000, 0.000, 0.667,
    0.000, 0.000, 0.833,
    0.000, 0.000, 1.000,
    0.000, 0.000, 0.000,
    0.143, 0.143, 0.143,
    0.286, 0.286, 0.286,
    0.429, 0.429, 0.429,
    0.571, 0.571, 0.571,
    0.714, 0.714, 0.714,
    0.857, 0.857, 0.857,
    1.000, 1.000, 1.000
]

_BLACK = (0, 0, 0)
_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)

_COLOR1 = tuple(255 * x for x in (0.000, 0.447, 0.741))


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def rect_with_opacity(image, top_left, bottom_right, fill_color, fill_opacity):
    with_fill = image.copy()
    with_fill = cv2.rectangle(with_fill, top_left, bottom_right, fill_color,
                              cv2.FILLED)
    return cv2.addWeighted(with_fill, fill_opacity, image, 1 - fill_opacity, 0,
                           image)


def vis_mask(image, mask, alpha=0.5, color=None):
    colmap = (np.array(pascal_colormap) * 255).round().astype("uint8").reshape(256, 3)

    if color is None:
        color = detectron_colormap[np.random.choice(len(detectron_colormap))][::-1]
    else:
        while color >= 255:
            color = color - 254
        color = colmap[color]

    im = np.where(np.repeat((mask > 0)[:, :, None], 3, axis=2),
                  image * (1 - alpha) + color * alpha, image)
    im = im.astype('uint8')
    return im


def vis_bbox(image,
             box,
             border_color=_BLACK,
             fill_color=_COLOR1,
             fill_opacity=0.65,
             thickness=2):
    """Visualizes a bounding box."""
    x0, y0, w, h = box
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    # Draw border
    if fill_opacity > 0 and fill_color is not None:
        image = rect_with_opacity(image, (x0, y0), (x1, y1), tuple(fill_color),
                                  fill_opacity)
    image = cv2.rectangle(image, (x0, y0), (x1, y1), tuple(border_color),
                          thickness)
    return image


def bbox_from_mask(mask, order='Y1Y2X1X2', return_none_if_invalid=False):
    reduced_y = np.any(mask, axis=0)
    reduced_x = np.any(mask, axis=1)

    x_min = reduced_y.argmax()
    if x_min == 0 and reduced_y[0] == 0:  # mask is all zeros
        if return_none_if_invalid:
            return None
        else:
            return -1, -1, -1, -1

    x_max = len(reduced_y) - np.flip(reduced_y, 0).argmax()

    y_min = reduced_x.argmax()
    y_max = len(reduced_x) - np.flip(reduced_x, 0).argmax()

    if order == 'Y1Y2X1X2':
        return y_min, y_max, x_min, x_max
    elif order == 'X1X2Y1Y2':
        return x_min, x_max, y_min, y_max
    elif order == 'X1Y1X2Y2':
        return x_min, y_min, x_max, y_max
    elif order == 'Y1X1Y2X2':
        return y_min, x_min, y_max, x_max
    else:
        raise ValueError("Invalid order argument: %s" % order)


def overlay_mask_on_image(image, mask, mask_opacity=0.6, mask_colour=(0, 255, 0)):
    if mask.ndim == 3:
        assert mask.shape[2] == 1
        _mask = mask.squeeze(axis=2)
    else:
        _mask = mask
    mask_bgr = np.stack((_mask, _mask, _mask), axis=2)
    masked_image = np.where(mask_bgr > 0, mask_colour, image)
    return ((mask_opacity * masked_image) + ((1. - mask_opacity) * image)).astype(np.uint8)


def annotate_instance(image, colour, mask=None, points=None, text_label=None, draw_box=False):
    """
    :param image: np.ndarray(H, W, 3)
    :param color: tuple/list(int, int, int) in range [0, 255]
    :param mask: np.ndarray(H, W)
    :param points: list[tuples]
    :param text_label: str
    :param draw_box: bool
    :return: np.ndarray(H, W, 3)
    """
    assert mask is not None or points is not None
    if mask is not None:
        assert image.shape[:2] == mask.shape, "Shape mismatch between image {} and mask {}".format(image.shape,
                                                                                                   mask.shape)
    colour = tuple(int(_c) for _c in colour)

    if mask is not None:
        overlayed_image = overlay_mask_on_image(image, mask, mask_colour=colour)
        if draw_box:
            bbox = bbox_from_mask(mask, order='X1Y1X2Y2', return_none_if_invalid=True)
            if not bbox:
                return overlayed_image
            xmin, ymin, xmax, ymax = bbox
            cv2.rectangle(overlayed_image, (xmin, ymin), (xmax, ymax), color=colour, thickness=2)
    elif points is not None:
        overlayed_image = image.astype(np.float32)
        for point in points:
            cv2.circle(overlayed_image, (int(point[0]), int(point[1])), radius=3, color=colour, thickness=-1)

    # print(overlayed_image.shape)
    # print("Label: ", text_label)
    # print((xmin, ymin))
    if text_label is not None:
        cv2.putText(overlayed_image, str(text_label), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    return overlayed_image


def overlay_prob(image, prob_map, alpha=0.4):
    """

    :param image: numpy array of shape H x W x 3
    :param prob_map: probability map of shape I x H x W
    :param normalise:
    :return:
    """
    assert image.shape[:2] == prob_map.shape[-2:]
    # ignore background
    prob = prob_map[1:].max(axis=0)

    image = image.astype('float') / image.max()
    prob = prob / prob.max()

    prob_colour = cv2.applyColorMap((prob * 255).astype('uint8'), cv2.COLORMAP_JET)
    img_overlay = cv2.addWeighted(image, alpha, prob_colour.astype('float') / 255.0, 1 - alpha, 0.0)

    return img_overlay


# optional mapping of values with morphological shapes
def morph_shape(val):
    if val == 0:
        return cv2.MORPH_RECT
    elif val == 1:
        return cv2.MORPH_CROSS
    elif val == 2:
        return cv2.MORPH_ELLIPSE


def erosion(mask):
    kernel = np.ones((15, 15), np.uint8)
    eroded_mask = np.stack([cv2.erode(_m, kernel, iterations=1) for _m in mask], axis=0)
    return eroded_mask


def dilation(mask):
    # dilation_shape = morph_shape(cv.getTrackbarPos(title_trackbar_element_shape, title_dilation_window))
    # element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
    #                                    (dilatation_size, dilatation_size))
    kernel = np.ones((15, 15), np.uint8)
    dilated_mask = np.stack([cv2.dilate(_m, kernel, iterations=1) for _m in mask], axis=0)
    return dilated_mask


def remove_boundary(prob):
    res = []
    for _p in prob:
        mask = (_p > 0.5).astype('uint8')
        e_mask = erosion(mask)
        d_mask = dilation(mask)
        boundary = ((d_mask - e_mask) > 0).astype('uint8')
        inv = 1 - boundary
        res.append(_p * inv)

    return np.stack(res, axis=0)
