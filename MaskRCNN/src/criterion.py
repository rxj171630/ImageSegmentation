import torch
from torch import nn
from torch.nn import functional as F
from src.utils import *

class RPNLoss(nn.Module):
    def __init__(self, l=10, batch_size=256) -> None:
        super().__init__()
        self.l = l
        self.batch_size = batch_size
    
    def get_obj_loss(self, obj_score, target):
        '''
        @params:
            obj_score: pytorch tensor of size (N) containing object scores for N proposal boxes
            target: pytorch tensor of size (N) containing ground truth scores for N proposal boxes
        
        @return:
            loss: object score loss
        '''
        return F.binary_cross_entropy_with_logits(obj_score, target.detach(), reduction="mean")
    
    def get_reg_loss(self, reg_score, target_reg_score):
        '''
        @params:
            reg_score: tensor containing bbox regression scores. (N, 4)
            target_reg_score: tensor ground truth regression scores. (N, 4)

        @return:
            loss: regression score loss
        '''
        return F.smooth_l1_loss(reg_score, target_reg_score.detach(), reduction="sum")
    
    def forward(self, reg_scores, obj_scores, pred_bboxes, target_bboxes, anchors):
        '''
        @params:
            reg_scores: tensor regression score of each anchors. (N, 4)
            obj_scores: tensor object score of each anchor. (N)
            pred_bboxes: tensor predicted bboxes in xyxy coords. (N, 4)
            target_bboxes: xyxy coordinates of ground truth bboxes (Nbox, 4)
            anchors: tensor anchor boxes in xywh coords. (N, 4)
        '''
        target_xywh = xyxy2xywh(target_bboxes)
        

        ### Assign Positive/Negative labels
        ious = jaccard(pred_bboxes, target_bboxes)
        mask_pred_boxes = ious > 0.7
        for i in range(target_bboxes.shape[0]):
            if mask_pred_boxes[:, i].sum() == 0:
                mask_pred_boxes[ious[:, i].argmax(), i] = True
        
        positives = mask_pred_boxes.nonzero()
        negatives = (~mask_pred_boxes).nonzero()

        ### Select positive/negative samples for minibatch
        n_pos = self.batch_size/2
        n_neg = self.batch_size/2
        if positives.shape[0] < n_pos:
            n_pos = positives.shape[0]
            n_neg += self.batch_size/2 - n_pos

        n_pos = int(n_pos)
        n_neg = int(n_neg)

        idx_pos = torch.ones(positives.shape[0],device=target_bboxes.device).multinomial(n_pos)
        # idx_pos = torch.randint(high=positives.shape[0], size=(n_pos), device=target_bboxes.device)
        idx_neg = torch.ones(negatives.shape[0],device=target_bboxes.device).multinomial(n_neg)
        # idx_neg = torch.randint(high=negatives.shape[0], size=(n_neg), device=target_bboxes.device)
        sample_pos_id = positives[idx_pos]
        sample_neg_id = negatives[idx_neg]
        
        ### Get regression scores for positive samples
        reg_scores_pos = reg_scores[sample_pos_id[:, 0]]

        ### Calculate ground truth regression scores corresponding to positive samples
        anchors_pos = anchors[sample_pos_id[:, 0]]
        target_xywh_pos = target_xywh[sample_pos_id[:, 1]]
        target_reg_scores = torch.zeros_like(target_xywh_pos)
        target_reg_scores[:, :2] = (target_xywh_pos[:, :2] - anchors_pos[:, :2]) / anchors_pos[:, 2:]
        target_reg_scores[:, 2:] = torch.log(target_xywh_pos[:, 2:] / anchors_pos[:, 2:])

        ### Calculate regression loss
        loss_reg = self.get_reg_loss(reg_scores_pos, target_reg_scores.detach()) * (self.l / 2500)

        ### Get object scores for positive and negative samples
        obj_scores_pos = obj_scores[sample_pos_id[:, 0]]
        obj_scores_neg = obj_scores[sample_neg_id[:, 0]]
        sample_obj_scores = torch.cat([obj_scores_pos, obj_scores_neg])

        ### Construct ground truth object scores for corresponding samples 
        target_obj_scores = torch.cat((torch.ones_like(obj_scores_pos), torch.zeros_like(obj_scores_neg)))

        ### Calculate object loss
        loss_obj = self.get_obj_loss(sample_obj_scores, target_obj_scores.detach())

        return loss_reg + loss_obj






        
        




