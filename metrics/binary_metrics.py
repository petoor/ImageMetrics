import numpy as np
from skimage.measure import label
from skimage.metrics import hausdorff_distance

class BinaryImageMetrics():
    def __init__(self, y_true, y_pred):
        #TODO: Add documentation
        #TODO: implement Adjusted Rand Index and https://web.stanford.edu/class/cs273/scribing/2004/class8/scribe8.pdf
        
        # Numpy have the bug : ValueError: cannot set WRITEABLE flag to True of this array
        # That is why we copy the array

        # We force the predictions to be binary

        y_true = np.copy(y_true)
        y_pred = np.copy(y_pred)
        y_true[y_true>0]=1
        y_pred[y_pred>0] = 1
        self.y_true= y_true
        self.y_pred = y_pred
        self.y_true_label = label(self.y_true)
        self.y_pred_label = label(self.y_pred)

    def count_ratio(self):
        count_y_true = np.max(self.y_true_label)
        count_y_pred = np.max(self.y_pred_label)
        if count_y_pred == 0 or count_y_true == 0:
            return 0.0
        else:
            return min(count_y_true/count_y_pred, count_y_pred/count_y_true)

    def f1(self, y_true=None, y_pred=None, obj=False, return_rectangle=False, blank_default_value=0.0):
        if obj:
            f1_obj = 0
            total_y_pred = max(np.bincount(self.y_pred.flatten(),minlength=2)[1],1)
            total_y_true = max(np.bincount(self.y_true.flatten(), minlength=2)[1],1)

            for idx in range(1, self.y_true_label.max()+1):
                gi, si = self._overlap(idx, return_rectangle=return_rectangle, y_true_p_switch=False)
                f1 = (np.bincount(gi.flatten(), minlength=2)[1]/total_y_true)*self.f1(y_true=gi, y_pred=si, obj=False)
                f1_obj += f1

            for idx in range(1, self.y_pred_label.max()+1):
                si, gi = self._overlap(idx, return_rectangle=return_rectangle, y_true_p_switch=True)
                f1 = (np.bincount(si.flatten(), minlength=2)[1]/total_y_pred)*self.f1(y_true=si, y_pred=gi, obj=False)
                f1_obj += f1

            f1_obj /= 2

            return f1_obj        
        else:
            if y_true is None:
                y_true = self.y_true
            if y_pred is None:
                y_pred = self.y_pred
                
            if y_true.max() == 0 and y_pred.max()==0:
                return blank_default_value 
            
            _, fn, fp, tp = self.confusion_matrix(y_true, y_pred)
            f1 = 2*tp / (2*tp +fp + fn)
            return f1
        
    def iou(self, y_true=None, y_pred=None, obj=False, return_rectangle=False, blank_default_value=0.0):
        if obj:
            iou_obj = 0
            total_y_pred = max(np.bincount(self.y_pred.flatten(),minlength=2)[1],1)
            total_y_true = max(np.bincount(self.y_true.flatten(), minlength=2)[1],1)

            for idx in range(1, self.y_true_label.max()+1):
                gi, si = self._overlap(idx, return_rectangle=return_rectangle, y_true_p_switch=False)
                iou = (np.bincount(gi.flatten(), minlength=2)[1]/total_y_true)*self.iou(y_true=gi, y_pred=si, obj=False)
                iou_obj += iou

            for idx in range(1, self.y_pred_label.max()+1):
                si, gi = self._overlap(idx, return_rectangle=return_rectangle, y_true_p_switch=True)
                iou = (np.bincount(si.flatten(), minlength=2)[1]/total_y_pred)*self.iou(y_true=si, y_pred=gi, obj=False)
                iou_obj += iou

            iou_obj /= 2

            return iou_obj
    
        else:
            if y_true is None:
                y_true = self.y_true
            if y_pred is None:
                y_pred = self.y_pred
            
            if y_true.max() == 0 and y_pred.max()==0:
                return blank_default_value 
            
            _, fn, fp, tp = self.confusion_matrix(y_true, y_pred)
            iou = tp / (tp +fp + fn)
            return iou

    def mcc(self, y_true=None, y_pred=None, obj=False, return_rectangle=True, blank_default_value=0.0):
        if obj:
            mcc_obj = 0
            total_y_pred = max(np.bincount(self.y_pred.flatten(),minlength=2)[1],1)
            total_y_true = max(np.bincount(self.y_true.flatten(), minlength=2)[1],1)

            for idx in range(1, self.y_true_label.max()+1):
                gi, si = self._overlap(idx, return_rectangle=return_rectangle, y_true_p_switch=False)
                mcc = (np.bincount(gi.flatten(), minlength=2)[1]/total_y_true)*self.mcc(y_true=gi, y_pred=si, obj=False)
                mcc_obj += mcc

            for idx in range(1, self.y_pred_label.max()+1):
                si, gi = self._overlap(idx, return_rectangle=return_rectangle, y_true_p_switch=True)
                mcc = (np.bincount(si.flatten(), minlength=2)[1]/total_y_pred)*self.mcc(y_true=si, y_pred=gi, obj=False)
                mcc_obj += mcc

            mcc_obj /= 2
            
            return mcc_obj
    
        else:
            if y_true is None:
                y_true = self.y_true
            if y_pred is None:
                y_pred = self.y_pred
        
            if y_true.max() == 0 and y_pred.max()==0:
                return blank_default_value 
            
            tn, fn, fp, tp = self.confusion_matrix(y_true, y_pred)
            if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) == 0: # We can't divide by zero
                mcc = 0.0
            else:
                mcc = (tp*tn - fp*fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        return mcc

    def hausdorff_distance(self, y_true=None, y_pred=None, obj=False, return_rectangle=True, blank_default_value=None):
        if obj:
            haus_dist_obj = 0
            total_y_pred = np.bincount(self.y_pred.flatten(), minlength=2)[1]
            total_y_true = np.bincount(self.y_true.flatten(), minlength=2)[1]
        
            if total_y_true == 0 or total_y_pred == 0:
                # Note this conversion is different from
                # https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/evaluation/
                # But the Hausdorff distance is not defined for non-overlapping objects.
                # Note that the distance of sqrt(h^2+w^2) is an arbary choice.
                # 0.0 would skew the distance to much to the positive side, and
                # infinite is misleading. 
                # In case both objects are empty, we return the max dist of y_true
                if blank_default_value is not None:
                    return blank_default_value
                else:
                    return np.sqrt(self.y_true.shape[0]**2+self.y_true.shape[1]**2)

            for idx in range(1, self.y_true_label.max()+1):
                gi, si = self._overlap(idx, return_rectangle=return_rectangle, y_true_p_switch=False)
                if gi.max() == 0 or si.max() == 0:
                    haus_dist = ((gi.shape[0]*gi.shape[1])/total_y_true)*np.sqrt(gi.shape[0]**2+gi.shape[1]**2)
                else:
                    haus_dist = (np.bincount(gi.flatten())[1]/total_y_true)*self.hausdorff_distance(y_true=gi, y_pred=si, obj=False)
                haus_dist_obj += haus_dist

            for idx in range(1, self.y_pred_label.max()+1):
                si, gi = self._overlap(idx, return_rectangle=return_rectangle, y_true_p_switch=True)
                if gi.max() == 0 or si.max() == 0:
                    haus_dist = ((si.shape[0]*si.shape[1])/total_y_pred)*np.sqrt(si.shape[0]**2+si.shape[1]**2)
                else:
                    haus_dist = (np.bincount(si.flatten())[1]/total_y_pred)*self.hausdorff_distance(y_true=si, y_pred=gi, obj=False)
                haus_dist_obj += haus_dist

            haus_dist_obj /= 2
            return haus_dist_obj

        else:
            if y_true is None:
                y_true = self.y_true
            if y_pred is None:
                y_pred = self.y_pred
       
            if y_true.max() == 0 or y_pred.max() == 0:
                haus_dist = np.sqrt(y_true.shape[0]*y_true.shape[1])
            else:
                haus_dist = hausdorff_distance(y_true, y_pred)
            return haus_dist
                     

    def confusion_matrix(self, y_true, y_pred):
        y_true= y_true.flatten()
        y_pred = y_pred.flatten()*2
        cm = y_true+y_pred
        cm = np.bincount(cm, minlength=4)
        tn, fn, fp, tp = np.float64(cm)
        return tn, fn, fp, tp
    
    def _overlap(self, idx, return_rectangle=False, y_true_p_switch=False):
        if y_true_p_switch:
            y_pred = self.y_true_label
            y_true = self.y_pred_label
        else:
            y_true = self.y_true_label
            y_pred = self.y_pred_label

        roi_y_true = np.argwhere(y_true == idx)
        roi_y_pred = y_pred[roi_y_true[:,0],roi_y_true[:,1]]

        # Finds max overlap, excluding background
        matching_idx = np.bincount(roi_y_pred)
        if len(matching_idx)<=1:
            matching_idx = -1
        else:
            matching_idx = matching_idx[1:].argmax()+1
        if return_rectangle:
            bbox, _, _ = self._bounding_box(roi_y_true, at_origon=False)
            roi_y_true = y_true[bbox[0]:bbox[2]+1,bbox[1]:bbox[3]+1]
            roi_y_pred = y_pred[bbox[0]:bbox[2]+1,bbox[1]:bbox[3]+1]
            roi_y_true =(roi_y_true==idx)*1

        else:
            roi_y_true = np.ones((len(roi_y_true)),dtype=int)
        roi_y_pred = (roi_y_pred==matching_idx)*1
        
        return roi_y_true, roi_y_pred
        
    def _bounding_box(self, points, at_origon=False):
        bbox = [min(points[:,0]), min(points[:,1]), max(points[:,0]), max(points[:,1])]
        min_x = bbox[0]
        min_y = bbox[1]
        if at_origon:
            bbox = [bbox[0] - min_x,
                    bbox[1] - min_y,
                    bbox[2] - min_x,
                    bbox[3] - min_y]
        return bbox, min_x, min_y
        
