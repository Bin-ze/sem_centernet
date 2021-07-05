import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
import torch.nn as nn

@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 semantic_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)
        backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.semantic_head=build_head(semantic_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv=[]
        self.conv1=nn.Conv2d(256,256,1)
        self.conv2 = nn.Conv2d(512, 256, 1)
        self.conv3 = nn.Conv2d(1024, 256, 1)
        self.conv4 = nn.Conv2d(2048, 256, 1)
        self.conv.append(self.conv1)
        self.conv.append(self.conv2)
        self.conv.append(self.conv3)
        self.conv.append(self.conv4)
        self.tran_feature = self._build_trans(64,64,80)

    def _build_trans(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        sem = self.backbone(img)
        if self.with_neck:
            det = self.neck(sem)
        return sem,det

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_semantic_seg,
                      gt_m=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        sem,det = self.extract_feat(img)
        sem_input=[]
        for i,sem_featue in enumerate(sem):
            sem_input.append(self.conv[i](sem_featue))
        sem1=tuple(sem_input)

        sem_loss,mask_pred,semantic_feat=self.semantic_head.forward_train(sem1, gt_semantic_seg)
#mask_transform:
        with torch.no_grad():
            probs = torch.sigmoid(mask_pred)
            a = torch.where(probs > 0.5, 1.0,0.5)
            a1 = a[:, 1:12, :, :]
            a2 = a[:, 13:26, :, :]
            a3 = a[:, 27:29, :, :]
            a4 = a[:, 31:45, :, :]
            a5 = a[:, 46:66, :, :]
            a6 = a[:, 67:68, :, :]
            a7 = a[:, 70:71, :, :]
            a8 = a[:, 72:83, :, :]
            a9 = a[:, 84:91, :, :]
            mask = torch.cat((a1, a2, a3, a4, a5, a6, a7, a8, a9), dim=1)
        det_feature=semantic_feat+det[0]
        det_feature=(self.tran_feature(det_feature)*mask,)
        losses = self.bbox_head.forward_train(det_feature ,img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        losses.update(sem_loss)

        return losses


    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time au:gmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        sem,feat = self.extract_feat(img)
        sem_input = []
        for i, sem_featue in enumerate(sem):
            sem_input.append(self.conv[i](sem_featue))
        sem1 = tuple(sem_input)

        mask_pred, semantic_feat = self.semantic_head.forward(sem1)
        #mask_transform:
        with torch.no_grad():
            probs = torch.sigmoid(mask_pred)
            a = torch.where(probs > 0.5, 1.0, 0.5)
            a1 = a[:, 1:12, :, :]
            a2 = a[:, 13:26, :, :]
            a3 = a[:, 27:29, :, :]
            a4 = a[:, 31:45, :, :]
            a5 = a[:, 46:66, :, :]
            a6 = a[:, 67:68, :, :]
            a7 = a[:, 70:71, :, :]
            a8 = a[:, 72:83, :, :]
            a9 = a[:, 84:91, :, :]
            mask = torch.cat((a1, a2, a3, a4, a5, a6, a7, a8, a9), dim=1)
        det=semantic_feat+feat[0]
        det_feature=(self.tran_feature(det)*mask,)
        results_list = self.bbox_head.simple_test(
            det_feature, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape
        # TODO:move all onnx related code in bbox_head to onnx_export function
        det_bboxes, det_labels = self.bbox_head.get_bboxes(*outs, img_metas)

        return det_bboxes, det_labels
