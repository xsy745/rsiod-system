"""
"""
# @author: xsy745
# @createTime:


import os.path

import cv2
import torch
import numpy as np
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend


def preprocess_warpAffine(image, dst_width=1024, dst_height=1024):
    """
    对输入图像进行仿射变换，调整图像大小，转换颜色通道，归一化像素值，最后转为 PyTorch 张量。同时返回逆仿射变换矩阵。
    """
    # 计算缩放比例，取宽度和高度缩放比例中的较小值，以确保图像在目标尺寸内完整显示
    scale = min((dst_width / image.shape[1], dst_height / image.shape[0]))
    # 计算图像在目标宽度方向上的偏移量
    ox = (dst_width - scale * image.shape[1]) / 2
    # 计算图像在目标高度方向上的偏移量
    oy = (dst_height - scale * image.shape[0]) / 2
    # 定义仿射变换矩阵 M，它是一个 2x3 的矩阵，用于仿射变换操作
    # 矩阵的左上角 2x2 子矩阵控制缩放，最后一列控制平移
    M = np.array([
        [scale, 0, ox],
        [0, scale, oy]
    ], dtype=np.float32)
    # 使用 cv2.warpAffine 函数对输入图像进行仿射变换
    # image: 输入图像
    # M: 仿射变换矩阵
    # (dst_width, dst_height): 输出图像的尺寸
    # flags=cv2.INTER_LINEAR: 插值方法为线性插值
    # borderMode=cv2.BORDER_CONSTANT: 边界填充模式为常数填充
    # borderValue=(114, 114, 114): 填充的常数值为灰色 (114, 114, 114)
    img_pre = cv2.warpAffine(image, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    # 计算仿射变换矩阵 M 的逆矩阵 IM，用于后续将检测框坐标映射回原始图像
    IM = cv2.invertAffineTransform(M)
    # 将 BGR 转换为 RGB，将像素值归一化到 [0, 1] 范围
    img_pre = (img_pre[..., ::-1] / 255.0).astype(np.float32)
    # 将 (H, W, C) 转换为 (1, C, H, W)
    img_pre = img_pre.transpose(2, 0, 1)[None]
    img_pre = torch.from_numpy(img_pre)
    return img_pre, IM


def xywhr2xyxyxyxy(center):
    """
    五参数表示法转换成八参数表示法
    reference: https://github.com/ultralytics/ultralytics/blob/v8.1.0/ultralytics/utils/ops.py#L545
    """
    is_numpy = isinstance(center, np.ndarray)
    cos, sin = (np.cos, np.sin) if is_numpy else (torch.cos, torch.sin)

    ctr = center[..., :2]
    w, h, angle = (center[..., i: i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = np.concatenate(vec1, axis=-1) if is_numpy else torch.cat(vec1, dim=-1)
    vec2 = np.concatenate(vec2, axis=-1) if is_numpy else torch.cat(vec2, dim=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return np.stack([pt1, pt2, pt3, pt4], axis=-2) if is_numpy else torch.stack([pt1, pt2, pt3, pt4], dim=-2)


def probiou(obb1, obb2, eps=1e-7):
    """
    Calculate the prob iou between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.
    """
    def covariance_matrix(obb):
        # Extract elements
        w, h, r = obb[2:5]
        a = (w ** 2) / 12
        b = (h ** 2) / 12

        cos_r = torch.cos(torch.tensor(r))
        sin_r = torch.sin(torch.tensor(r))

        # Calculate covariance matrix elements
        a_val = a * cos_r ** 2 + b * sin_r ** 2
        b_val = a * sin_r ** 2 + b * cos_r ** 2
        c_val = (a - b) * sin_r * cos_r

        return a_val, b_val, c_val

    a1, b1, c1 = covariance_matrix(obb1)
    a2, b2, c2 = covariance_matrix(obb2)

    x1, y1 = obb1[:2]
    x2, y2 = obb2[:2]

    t1 = ((a1 + a2) * ((y1 - y2) ** 2) + (b1 + b2) * ((x1 - x2) ** 2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)
    t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)
    t3 = torch.log(((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2) / (
                4 * torch.sqrt(a1 * b1 - c1 ** 2) * torch.sqrt(a2 * b2 - c2 ** 2) + eps) + eps)

    bd = 0.25 * t1 + 0.5 * t2 + 0.5 * t3
    hd = torch.sqrt(1.0 - torch.exp(-torch.clamp(bd, eps, 100.0)) + eps)
    return 1 - hd


def NMS(boxes, iou_thres):
    remove_flags = [False] * len(boxes)

    keep_boxes = []
    for i, ibox in enumerate(boxes):
        if remove_flags[i]:
            continue

        keep_boxes.append(ibox)
        for j in range(i + 1, len(boxes)):
            if remove_flags[j]:
                continue

            jbox = boxes[j]
            if (ibox[6] != jbox[6]):
                continue
            if probiou(ibox, jbox) > iou_thres:
                remove_flags[j] = True
    return keep_boxes


def postprocess(pred, IM=[], conf_thres=0.25, iou_thres=0.45):
    """
    输入是模型推理的结果，即21504个预测框
    1,21504,20 [cx,cy,w,h,class*15,rotated]
    """
    boxes = []
    for item in pred[0]:
        cx, cy, w, h = item[:4]
        angle = item[-1]
        label = item[4:-1].argmax()
        confidence = item[4 + label]
        if confidence < conf_thres:
            continue
        boxes.append([cx, cy, w, h, angle, confidence, label])

    boxes = np.array(boxes)
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    wh = boxes[:, 2:4]
    boxes[:, 0] = IM[0][0] * cx + IM[0][2]
    boxes[:, 1] = IM[1][1] * cy + IM[1][2]
    boxes[:, 2:4] = IM[0][0] * wh
    boxes = sorted(boxes.tolist(), key=lambda x: x[5], reverse=True)

    return NMS(boxes, iou_thres)


def hsv2bgr(h, s, v):
    """
    将 HSV 颜色空间转换为 BGR 颜色空间
    """
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    r, g, b = 0, 0, 0

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q

    return int(b * 255), int(g * 255), int(r * 255)


def random_color(id):
    """
    根据输入的 ID 生成随机颜色
    """
    h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
    s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
    return hsv2bgr(h_plane, s_plane, 1)


def object_info_to_str(data) -> str:
    """
    将目标信息转换为字符串
    """
    transformed_data = []
    transformed_data_str = ''
    for item in data:
        # 提取第一个元素（如 "plane"）
        transformed_item = [item[0]]
        # 遍历数组中的每一行
        for row in item[1]:
            # 将数组行转换为列表
            transformed_item.append(row.tolist())
        transformed_item.append(item[2])
        transformed_data.append(transformed_item)
    for item in transformed_data:
        transformed_data_str += str(item) + '\n'
    return transformed_data_str


def infer(model_dir, image_dir):

    image_name_extension = os.path.basename(image_dir)
    image_name = os.path.splitext(image_name_extension)[0]
    image = cv2.imread(image_dir)

    img_pre, IM = preprocess_warpAffine(image)
    model = AutoBackend(weights=model_dir)
    names = model.names
    # 1,21504,20
    result = model(img_pre)[0].transpose(-1, -2)

    boxes = postprocess(result, IM)
    confs = [box[5] for box in boxes]
    classes = [int(box[6]) for box in boxes]
    boxes = xywhr2xyxyxyxy(np.array(boxes)[..., :5])

    object_info = []

    for i, box in enumerate(boxes):
        confidence = confs[i]
        label = classes[i]
        color = random_color(label)
        cv2.polylines(image, [np.asarray(box, dtype=int)], True, color, 2)
        caption = f"{names[label]} {confidence:.2f}"
        w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
        left, top = [int(b) for b in box[0]]
        cv2.rectangle(image, (left - 3, top - 33), (left + w + 10, top), color, -1)
        cv2.putText(image, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)

        object_info.append([names[label], box, confidence])

    # model_name_extension = os.path.basename(model_dir)
    # model_name = os.path.splitext(model_name_extension)[0]
    # detected_image_name = image_name + '_' + model_name + '_' + 'predict.jpg'

    detected_image_name = image_name + '_' + 'predict.jpg'
    detected_image_dir = os.path.dirname(image_dir) + '/' + detected_image_name
    cv2.imwrite(detected_image_dir, image)

    return detected_image_dir, object_info_to_str(object_info)


if __name__ == "__main__":
    model_dir = '../models/yolov8n-obb.pt'
    image_dir = '../images/P0021.jpg'

    detected_image_dir,detected_object_info = infer(model_dir, image_dir)
    print(detected_object_info)


