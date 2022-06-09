import torch
import cv2
from src.pascal_voc import VOC_CLASSES
import matplotlib.pyplot as plt

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A U B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def xyxy2xywh(xyxy):
    '''
    @params:
        xyxy: coordinates in xyxy (N, 4)
    
    @return:
        xywh: coordinates in xywh (N, 4)
    '''
    xywh = torch.zeros_like(xyxy)
    xywh[:, :2] = 0.5 * (xyxy[:, 2:] + xyxy[:, :2])
    xywh[:, 2:] = xyxy[:, 2:] - xyxy[:, :2]

    return xywh

def xywh2xyxy(xywh):
        '''
        @params:
            xywh: xywh coordinate in torch array of size (N, 4)

        @return:
            xyxy: xyxy coordinate in torch array of size (N, 4)
        '''

        xyxy = torch.zeros_like(xywh)
        xyxy[:, :2] = xywh[:, :2] - 0.5 * xywh[:, 2:]
        xyxy[:, 2:] = xywh[:, :2] + 0.5 * xywh[:, 2:]

        return xyxy

def draw_bboxes(image, label, bbox):
    image = image.permute(1, 2, 0).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(bbox)
    for i in range(bbox.shape[0]):
        box = bbox[i]
        p1 = (int(box[0] * image.shape[1]), int(box[1] * image.shape[0]))
        p2 = (int(box[2] * image.shape[1]), int(box[3] * image.shape[0]))
        cv2.rectangle(image, p1, p2, color=[128, 0, 0], thickness=2)
        text_size, baseline = cv2.getTextSize(VOC_CLASSES[label[i]], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (p1[0], p1[1] - text_size[1])
        cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]),
                    [128, 0, 0], -1)

        cv2.putText(image, VOC_CLASSES[label[i]], (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

    plt.figure(figsize = (15,15))
    plt.imshow(image)