import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead_clips_flow
from mmseg.models.utils import *

from mmcv.utils import Registry, build_from_cfg
from torch import nn
from torch.nn import functional as F

from .mask2former.criterion_norm import VideoSetCriterion
from .mask2former.matcher_norm import VideoHungarianMatcher


@HEADS.register_module()
class THEMaskHead(BaseDecodeHead_clips_flow):

    def __init__(
        self,
        input_shape: dict,
        in_index: list,
        *,
        num_classes: int,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        dataset='',
        in_features,
        loss_cfg,
        pixel_decoder_cfg,
        transformer_decoder_cfg,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        in_channels = [v['channels'] for k, v in input_shape.items()]
        super().__init__(
            input_transform='multiple_select',
            in_channels=in_channels,
            channels=256,
            num_classes=num_classes,
            in_index=in_index) #TODO channels not used here

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self._init_pixel_decoder(pixel_decoder_cfg, input_shape)
        self._init_transformer_decoder(transformer_decoder_cfg)

        self.num_classes = num_classes
        self._init_criterion(loss_cfg)

        self.in_features = in_features
        self.dataset = dataset


    def _init_transformer_decoder(self, transformer_decoder_cfg):
        transformer_in_feature = transformer_decoder_cfg.pop('transformer_in_feature')
        CONVS_DIM = transformer_decoder_cfg.pop('CONVS_DIM')
        MASK_DIM = transformer_decoder_cfg.pop('MASK_DIM')
        if transformer_in_feature == "transformer_encoder":
            transformer_predictor_in_channels = CONVS_DIM
        elif transformer_in_feature == "pixel_embedding":
            transformer_predictor_in_channels = MASK_DIM
        elif transformer_in_feature == "multi_scale_pixel_decoder":  # for maskformer2
            transformer_predictor_in_channels = CONVS_DIM
        
        self.predictor = build_from_cfg(
            transformer_decoder_cfg, 
            HEADS, 
            dict(
                in_channels=transformer_predictor_in_channels,
                mask_classification=True
            )
        )
        self.transformer_in_feature = transformer_in_feature


    def _init_pixel_decoder(self, pixel_decoder_cfg, input_shape):
        self.pixel_decoder = build_from_cfg(pixel_decoder_cfg, HEADS, dict(input_shape=input_shape))


    def _init_criterion(self, loss_cfg):
        matcher_cfg = loss_cfg['matcher_cfg']
        matcher = VideoHungarianMatcher(cost_class=matcher_cfg['class_weight'], cost_mask = matcher_cfg['mask_weight'], cost_dice = matcher_cfg['dice_weight'], num_points=loss_cfg['train_num_points'])

        weight_dict = {"loss_ce": matcher_cfg.get('class_weight'), 
                    "loss_mask": matcher_cfg.get('mask_weight'), 
                    "loss_dice": matcher_cfg.get('dice_weight'), 
                    "loss_ce_unmatched": matcher_cfg.get('class_weight_unmatched'), 
                    "loss_mask_unmatched": matcher_cfg.get('mask_weight_unmatched'), 
                    "loss_dice_unmatched": matcher_cfg.get('dice_weight_unmatched')}

        if loss_cfg['deep_supervision']:
            dec_layers = loss_cfg['dec_layers']
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        self.criterion = VideoSetCriterion(
            self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=loss_cfg['no_object_weight'],
            losses=losses,
            num_points=loss_cfg['train_num_points'],
            oversample_ratio=loss_cfg['oversample_ratio'],
            importance_sample_ratio=loss_cfg['importance_sample_ratio'],
            round_weight=loss_cfg['round_weight'],
            aux_weight=loss_cfg['aux_weight'],
        )
    

    def forward(self, features, mask=None, batch_size=None, num_clips=None, imgs=None):
        if self.training:
            outputs, outputs_aux = self.layers(features, mask)
            return outputs, outputs_aux
        else:
            outputs, outputs_aux = self.layers(features, mask)
            mask_cls_results = outputs["pred_logits"] 
            mask_pred_results = outputs["pred_masks"] 
            pred_mq = self.semantic_inference(mask_cls_results, mask_pred_results)
            mask_cls_results_aux = outputs_aux["pred_logits"] 
            mask_pred_results_aux = outputs_aux["pred_masks"] 
            pred_subq = self.semantic_inference(mask_cls_results_aux, mask_pred_results_aux)

            del outputs, outputs_aux

            return pred_subq*0.5 + pred_mq*0.5


    def layers(self, features, mask=None):

        # tranform features from tuple into dict to match pixel_decoder format
        features_dict = {}
        for i in range(len(features)):
            features_dict[self.in_features[i]] = features[i]
        
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features_dict)
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(multi_scale_features, mask_features, mask)
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert (
                    transformer_encoder_features is not None
                ), "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(transformer_encoder_features, mask_features, mask)
            elif self.transformer_in_feature == "pixel_embedding":
                predictions = self.predictor(mask_features, mask_features, mask)
            else:
                predictions = self.predictor(features[self.transformer_in_feature], mask_features, mask)

        return predictions


    def match(self, outputs, outputs_aux, targets):
        outputs_match = {}
        outputs_match['pred_logits'] = outputs['pred_logits']
        b, q, t, h, w = outputs["pred_masks"].shape
        outputs_match['pred_masks'] = outputs_aux['pred_masks'].clone().view(b, t, q, h, w).permute(0,2,1,3,4)
        
        two_round_indices, two_round_indices_frame = self.criterion.matcher(outputs_match, targets)
        return two_round_indices, two_round_indices_frame


    def losses(self, outputs, outputs_aux, seg_label):
        tt_loss = {}
        
        targets, targets_frame = self.prepare_targets(seg_label)
        
        two_round_indices, two_round_indices_frame = self.match(outputs, outputs_aux, targets)
        
        two_round_indices_auxlist = []
        two_round_indices_auxlist_frame = []
        if "aux_outputs" in outputs:
            for i in range(len(outputs["aux_outputs"])):
                id1, id2 = self.match(outputs["aux_outputs"][i], outputs_aux["aux_outputs"][i], targets)
                two_round_indices_auxlist.append(id1)
                two_round_indices_auxlist_frame.append(id2)

        for idx, seg_logit in enumerate([outputs, outputs_aux]):

            if idx == 0:
                losses = self.criterion(seg_logit, targets, two_round_indices, two_round_indices_auxlist, idx)
            else:
                losses = self.criterion(seg_logit, targets_frame, two_round_indices_frame, two_round_indices_auxlist_frame, idx)

            for k in list(losses.keys()):
                if 'r0_' in k:
                    rmrd_k = k.replace('r0_', '')
                    rweight = self.criterion.round0_weight
                elif 'r1_' in k:
                    rmrd_k = k.replace('r1_', '')
                    rweight = self.criterion.round1_weight
                else: # FOR ce loss
                    rmrd_k = k
                    rweight = 1

                if rmrd_k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[rmrd_k] * rweight
                else:
                    # remove this loss if not specified in `weight_dict`
                    print(f'removing key:{k}')
                    losses.pop(k)
            
            if idx == 1:
                tt_loss.update({f'{k}_aux': self.criterion.aux_weight * v for k, v in losses.items()})
            else:
                tt_loss.update(losses)

        return tt_loss


    def prepare_targets(self, targets):
        new_targets = []
        new_targets_frame = []
        num_frames, _, h, w = targets.shape[-4:]

        for b in range(targets.shape[0]):
            video = targets[b].squeeze(1) # NF, H, W

            labels_per_video = video.unique() # contain background class
            # remove ignored region
            labels_per_video = labels_per_video[labels_per_video != 255] 

            padded_masks = torch.zeros((len(labels_per_video), num_frames, h, w), dtype=targets.dtype, device=targets.device)

            for i, label in enumerate(labels_per_video):
                idx = (video == label)
                padded_masks[i][idx]= 1

            new_targets.append(
                {
                    "labels": labels_per_video,
                    "masks": padded_masks,
                }
            )

            for frame in video:
                padded_masks_frame = torch.zeros((len(labels_per_video), 1, h, w), dtype=targets.dtype, device=targets.device)
                frame = frame.unsqueeze(0)
                for i, label in enumerate(labels_per_video):
                    idx_frame = (frame == label)
                    padded_masks_frame[i][idx_frame]= 1
                new_targets_frame.append(
                    {
                        "labels": labels_per_video,
                        "masks": padded_masks_frame,
                    }
                )

        return new_targets, new_targets_frame


    def semantic_inference(self, mask_cls, mask_pred):

        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqthw->btchw", mask_cls, mask_pred)
        semseg = semseg.flatten(0,1)

        return semseg
