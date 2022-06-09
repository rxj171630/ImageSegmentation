import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import nms
import torch.nn.functional as F
from src.detnet_fpn import detnet59_fpn
from src.utils import *

N_CLS = 20

class RPN(nn.Module):
    def __init__(self, backbone_path=None) -> None:
        super().__init__()

        self.training_mode = "train_rpn"
        self.backbone = detnet59_fpn(pretrained=backbone_path)
        
        '''
        Proposal scale: 32^2 on M2, 64^2 on M3, 128^2 on M4, 256^2 on M5.
        3 different aspect ratios:
            32:
                23*46
                32*32
                46*23
            64:
                45*90
                64*64
                90*45
            128:
                90*180
                128*128
                180*90
            256:
                180*360
                256*256
                360*180
        '''

        self.proposal_H = torch.tensor([23, 32, 46, 45, 64, 90, 90, 128, 180, 180, 256, 360], requires_grad=False)
        self.proposal_W = torch.tensor([46, 32, 23, 90, 64, 45, 180, 128, 90, 360, 256, 180], requires_grad=False)

        self.rpn_stem = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(256),
        )
        self.rpn_obj = nn.Conv2d(256, 3, 1)   # 2 obj score * 3 aspect ratio
        self.rpn_reg = nn.Conv2d(256, 12, 1)  # 4 reg * 3 aspect ratio

        ### Initialize weights
        for name, m in self.named_modules():
            if name.startswith("rpn"):
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0, 0.01)
                    if m.bias is not None:
                        m.bias.data.normal_(0, 0.01)

                if isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def save_model(self, path):
        torch.save({
            "rpn": self.state_dict(),
        }, path)
    
    def _get_anchors(self, reg_maps, H, W):
        anchor_H = self.proposal_H / H
        anchor_W = self.proposal_W / W

        anchors = []
        for i in range(4):
            xs = torch.arange(0, 1, 1 / reg_maps[i][0].shape[2], device=reg_maps[i].device)
            ys = torch.arange(0, 1, 1 / reg_maps[i][0].shape[1], device=reg_maps[i].device)
            xywh = torch.zeros_like(reg_maps[i][0])
            for j in range(3):
                xywh[4*j] = xs
                xywh[4*j+1] = ys.unsqueeze(1)
                xywh[4*j+2] = anchor_W[3*i+j]
                xywh[4*j+3] = anchor_H[3*i+j]
                pass

            anchors.append(xywh)

        return anchors

    def _reg2bbox(self, reg_maps, H, W, stride=1):
        '''
        @params:
            reg_maps: Regressor scores of rpn. Python list of regressor scores for 4 scales.
                    Each regressor score map is 12 * mapH * map W.
            
            H: Height of input image

            W: Width of input image.

            stride: rpn conv stride size.
        
        @return:
            xywh_maps: Python list of bounding boxes xywh maps for each regressor score map.
                    Each xywh maps is 12 * mapH * mapW
        '''
        bboxes = self._get_anchors(reg_maps, H, W)
        
        for i in range(4):
            xywh = bboxes[i]
            for j in range(3):
                xywh[4*j] = xywh[4*j] + xywh[4+j+2] * reg_maps[i][0][4*j]              # x = x_a + w_a * delta_x
                xywh[4*j+1] = xywh[4*j+1] + xywh[4*j+3] * reg_maps[i][0][4*j+1]        # y = y_a + h_a * delta_y
                xywh[4*j+2] = xywh[4*j+2] * torch.exp(reg_maps[i][0][4*j+2])           # w = w_a * exp(delta_w)
                xywh[4*j+3] = xywh[4*j+3] * torch.exp(reg_maps[i][0][4*j+3])           # h = h_a * exp(delta_h)
                pass
            pass

        return bboxes
    

    def _forward_rpn(self, feature_maps, H, W):
        rpn_features = [self.rpn_stem(m) for m in feature_maps]
        rpn_obj_scores = [self.rpn_obj(f) for f in rpn_features]    # 1*3*H*W
        rpn_reg_scores = [self.rpn_reg(f) for f in rpn_features]    # 1*12*H*W

        rpn_bboxes_xywh = self._reg2bbox(rpn_reg_scores, H, W)
        anchors = self._get_anchors(rpn_reg_scores, H, W)
        rpn_bboxes_xywh = torch.cat([e.view(12, -1).t().reshape(-1, 4) for e in rpn_bboxes_xywh])   # N*4
        anchors = torch.cat([e.view(12, -1).t().reshape(-1, 4) for e in anchors])                   # N*4
        rpn_reg_scores = torch.cat([e[0].view(12, -1).t().reshape(-1, 4) for e in rpn_reg_scores])     # N*4
        rpn_obj_scores = torch.cat([e[0].view(3, -1).t().flatten() for e in rpn_obj_scores])           # N

        rpn_bboxes_xyxy = xywh2xyxy(rpn_bboxes_xywh)

        within_boundary = torch.logical_and(rpn_bboxes_xyxy > 0, rpn_bboxes_xyxy < 1).min(dim=1)[0].detach()
        anchors = anchors[within_boundary]
        rpn_reg_scores = rpn_reg_scores[within_boundary]
        rpn_bboxes_xywh = rpn_bboxes_xywh[within_boundary]
        rpn_bboxes_xyxy = rpn_bboxes_xyxy[within_boundary]      # Discard all cross-boundary boxes
        rpn_obj_scores = rpn_obj_scores[within_boundary]        # Discard all cross-boundary boxes scores

        return rpn_reg_scores, anchors, rpn_bboxes_xywh, rpn_bboxes_xyxy, rpn_obj_scores

    def forward(self, x):
        # x: (1, 3, H, W), M: (1, 256, MH, MW)
        _, _, H, W = x.size()
        feature_maps = self.backbone(x)
        return self._forward_rpn(feature_maps, H, W)


class MaskRCNN(nn.Module):
    def __init__(self, backbone_path=None) -> None:
        super().__init__()

        self.training_mode = "train_rpn"
        self.backbone = detnet59_fpn(pretrained=backbone_path)
        
        '''
        Proposal scale: 32^2 on M2, 64^2 on M3, 128^2 on M4, 256^2 on M5.
        3 different aspect ratios:
            32:
                23*46
                32*32
                46*23
            64:
                45*90
                64*64
                90*45
            128:
                90*180
                128*128
                180*90
            256:
                180*360
                256*256
                360*180
        '''

        self.proposal_H = torch.tensor([23, 32, 46, 45, 64, 90, 90, 128, 180, 180, 256, 360], requires_grad=False)
        self.proposal_W = torch.tensor([46, 32, 23, 90, 64, 45, 180, 128, 90, 360, 256, 180], requires_grad=False)

        self.rpn_stem = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(256),
        )
        self.rpn_obj = nn.Conv2d(256, 6, 1)   # 2 obj score * 3 aspect ratio
        self.rpn_reg = nn.Conv2d(256, 12, 1)  # 4 reg * 3 aspect ratio

        ### Detection branch input size 256*7*7
        self.det_stem = nn.Sequential(
            nn.Linear(7*7*256, 1024, bias=False),
            nn.ELU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024, bias=False),
            nn.ELU(),
            nn.BatchNorm1d(1024),
        )
        self.det_cls = nn.Linear(1024, N_CLS + 1)
        self.det_reg = nn.Linear(1024, 4*N_CLS)


        ### Mask branch input size 256*14*14, output size N_CLS*28*28
        self.mask_layers = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, N_CLS, 3, padding=1),
        )

        ### Initialize weights
        for name, m in self.named_modules():
            if name.startswith(("rpn", "det", "mask")):
                if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                    m.weight.data.normal_(0, 0.01)
                    if m.bias is not None:
                        m.bias.data.normal_(0, 0.01)

                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


    def _anchor2bbox(self, reg_maps, H, W, stride=1):
        '''
        @params:
            reg_maps: Regressor scores of rpn. Python list of regressor scores for 4 scales.
                    Each regressor score map is 12 * mapH * map W.
            
            H: Height of input image

            W: Width of input image.

            stride: rpn conv stride size.
        
        @return:
            xywh_maps: Python list of bounding boxes xywh maps for each regressor score map.
                    Each xywh maps is 12 * mapH * mapW
        '''
        anchor_H = self.proposal_H / H
        anchor_W = self.proposal_W / W
        
        bboxes = []
        for i in range(4):
            xs = torch.arange(0, 1, 1 / reg_maps[i].shape[2], device=reg_maps[i].device)
            ys = torch.arange(0, 1, 1 / reg_maps[i].shape[1], device=reg_maps[i].device)
            xywh = torch.zeros_like(reg_maps[i])
            for j in range(3):
                xywh[4*j] = xs
                xywh[4*j+1] = ys.unsqueeze(1)
                xywh[4*j+2] = anchor_W[3*i+4*j]
                xywh[4*j+3] = anchor_H[3*i+4*j]

                xywh[4*j] = xywh[4*j] + xywh[4+j+2] * reg_maps[i][4*j]              # x = x_a + w_a * delta_x
                xywh[4*j+1] = xywh[4*j+1] + xywh[4*j+3] * reg_maps[i][4*j+1]        # y = y_a + h_a * delta_y
                xywh[4*j+2] = xywh[4*j+2] * torch.exp(reg_maps[i][4*j+2])           # w = w_a * exp(delta_w)
                xywh[4*j+3] = xywh[4*j+3] * torch.exp(reg_maps[i][4*j+3])           # h = h_a * exp(delta_h)
                pass

            bboxes.append(xywh)
            pass

        return bboxes
    
    def _xywh2xyxy(self, xywh):
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

    def _roi_align(self, feature_map, x1, y1, x2, y2, size=7):
        pass

    def _forward_rpn(self, feature_maps, H, W):
        rpn_features = [self.rpn_stem(m) for m in feature_maps]
        rpn_obj_scores = [self.rpn_obj(f) for f in rpn_features]    # 12*H*W
        rpn_reg_scores = [self.rpn_reg(f) for f in rpn_features]    # 6*H*W

        rpn_bboxes_xywh = self._anchor2bbox(rpn_reg_scores, H, W)
        rpn_bboxes_xywh = torch.cat([b.view(12, -1).t().reshape(-1, 4) for b in rpn_bboxes_xywh])   # N*4
        rpn_obj_scores = torch.cat([o.view(6, -1).t().reshape(-1, 2) for o in rpn_obj_scores])      # N*2

        rpn_bboxes_xyxy = self._xywh2xyxy(rpn_bboxes_xywh)

        within_boundary = torch.logical_and(rpn_bboxes_xyxy > 0, rpn_bboxes_xyxy < 1).min(dim=1)[0]
        rpn_bboxes_xywh = rpn_bboxes_xywh[within_boundary]
        rpn_bboxes_xyxy = rpn_bboxes_xyxy[within_boundary]      # Discard all cross-boundary boxes
        rpn_obj_scores = rpn_obj_scores[within_boundary]        # Discard all cross-boundary boxes scores

        return rpn_bboxes_xywh, rpn_bboxes_xyxy, rpn_obj_scores




    def forward(self, x):
        # x: (3, H, W), M: (256, MH, MW)
        _, H, W = x.size()
        feature_maps = self.backbone(x)
        bboxes_xywh, bboxes_xyxy, bboxes_obj = self._forward_rpn(feature_maps, H, W)


        if self.training:
            n_proposal = 2000
        else:
            n_proposal = 1000

    def set_training_mode(self, mode):
        if mode == "train_rpn":
            self.training_mode = "train_rpn"
        elif mode == "train_det":
            self.training_mode = "train_det"
        else:
            print("Invalid input. Work mode can be one of \"eval\", \"train_rpn\", or \"train_det\".")

    def freeze_backbone(self):
        for name, val in self.named_parameters():
            if name.startswith("backbone"):
                val.requires_grad = False
    
    def freeze_rpn(self):
        for name, val in self.named_parameters():
            if name.startswith("rpn"):
                val.requires_grad = False




        

