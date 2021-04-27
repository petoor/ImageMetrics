# This is the python code for calculating iou between pred_box and gt_box
# author:Forest 2019.7.19 # https://github.com/Treesfive/calculate-iou/blob/master/get_iou.py

import numpy as np
from skimage.measure import label

class BBoxMetrics():
    def __init__(self, y_true, y_pred, y_true_from_mask=True, y_pred_from_mask=True, iou_thresh=0.5):      
        # Numpy have the bug : ValueError: cannot set WRITEABLE flag to True of this array
        # That is why we copy the array

        # We force the predictions to be binary
        if y_true_from_mask:
            y_true = np.copy(y_true)
            y_true[y_true>0] = 1
            self.y_true_label = label(y_true)
            self.y_true_bbox = np.array(self._bounding_box(self.y_true_label))
        else:
            self.y_true_bbox = y_true

        if y_pred_from_mask:
            y_pred = np.copy(y_pred)
            y_pred[y_pred>0] = 1
            self.y_pred_label = label(y_pred)
            self.y_pred_bbox = np.array(self._bounding_box(self.y_pred_label))
        else:
            self.y_pred_bbox = y_pred
            
        self.tn, self.fn, self.fp, self.tp = self.confusion_matrix(iou_thresh)

    def get_iou(self, pred_box, gt_box):
        """
        pred_box : the coordinate for predict bounding box
        gt_box :   the coordinate for ground truth bounding box
        return :   the iou score
        the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
        the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
        """
        # 1.get the coordinate of inters
        ixmin = max(pred_box[0], gt_box[0])
        ixmax = min(pred_box[2], gt_box[2])
        iymin = max(pred_box[1], gt_box[1])
        iymax = min(pred_box[3], gt_box[3])

        iw = np.maximum(ixmax-ixmin+1., 0.)
        ih = np.maximum(iymax-iymin+1., 0.)

        # 2. calculate the area of inters
        inters = iw*ih

        # 3. calculate the area of union
        uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
            (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
            inters)

        # 4. calculate the overlaps between pred_box and gt_box
        iou = inters / uni

        return iou


    def get_max_iou(self, pred_boxes, gt_box):
        """
        calculate the iou multiple pred_boxes and 1 gt_box (the same one)
        pred_boxes: multiple predict  boxes coordinate
        gt_box: ground truth bounding  box coordinate
        return: the max overlaps about pred_boxes and gt_box
        """
        # 1. calculate the intersection coordinate
        if pred_boxes.shape[0] > 0:
            ixmin = np.maximum(pred_boxes[:, 0], gt_box[0])
            ixmax = np.minimum(pred_boxes[:, 2], gt_box[2])
            iymin = np.maximum(pred_boxes[:, 1], gt_box[1])
            iymax = np.minimum(pred_boxes[:, 3], gt_box[3])

            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)

        # 2.calculate the area of intersection
            inters = iw * ih

        # 3.calculate the area of union
            uni = ((pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.) +
                (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
                inters)

        # 4.calculate the overlaps and find the max overlap ,the max overlaps index for pred_box
            iou = inters / uni
            iou_max = np.max(iou)
            nmax = np.argmax(iou)
            return iou, iou_max, nmax

    def _bounding_box(self, label_img, at_origon=False):
        bbox = []
        for idx in range(1, np.max(label_img)+1):
            points = np.argwhere(label_img == idx)
            bbox.append([min(points[:,0]), min(points[:,1]), max(points[:,0]), max(points[:,1])])
        return bbox   

    def confusion_matrix(self, iou_thresh=0.5):
        assert iou_thresh >= 0.5 
        # Discussed in https://polyp.grand-challenge.org/Metrics/
        # With a IoU of 0.5 we can not have more than one prediction 
        # inside a ground truth region, thus multiple detections can not happen.
        tp = 0.0
        tn = 0.0
        fp = 0.0
        fn = 0.0

        if len(np.array(self.y_true_bbox)) == 0 and len(np.array(self.y_pred_bbox)) == 0:
            tn += 1.0

        elif len(np.array(self.y_true_bbox)) != 0 and len(np.array(self.y_pred_bbox)) == 0:
            fn += len(np.array(self.y_true_bbox))

        elif len(np.array(self.y_true_bbox)) == 0 and len(np.array(self.y_pred_bbox)) != 0:
            fp += len(np.array(self.y_pred_bbox))

        else:
            for pred_bbox in self.y_pred_bbox:
                _, iou_max, _ = self.get_max_iou(self.y_true_bbox, pred_bbox)
                if iou_max >= iou_thresh:
                    tp += 1.0
                else:
                    fp += 1.0
            
            for y_true in self.y_true_bbox:
                _, iou_max, _ = self.get_max_iou(self.y_pred_bbox, y_true)
                if iou_max >= iou_thresh:
                    pass # This is already handled in the tp logic since
                         # the iou score is communative J(A,B) = J(B,A) and we dont want double counts
                         # However, if this should be tp += 1 is debateable.
                else:
                    fn += 1.0
                  
        return tn, fn, fp, tp
