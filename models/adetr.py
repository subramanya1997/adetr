# copyright (c) Subramanya N. Licensed under the Apache License 2.0. All Rights Reserved
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .transformer import build_transformer
from utils.misc import NestedTensor, interpolate

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class ADETR(nn.Module):
    def __init__(
        self, backbone, transformer, num_classes, num_attributes, num_queries,
        aux_loss=False, contrastive_hdim=64, contrastive_loss=False,
        contrastive_align_loss=False
    ):
        """ Initializes the model.

        Args:
            backbone (nn.Module): torch modeule of the backbone to be used. See backbone.py
            transformer (nn.Module): torch module of the transformer to be used. See transformer.py
            num_classes (int): number of object classes
            num_queries (int): number of object queries, ie detection slot. This is the maximal
                number of objects ADETR can detect in a single image. For COCO, we recommend 100.
            aux_loss (bool): if True, computes auxiliary decoding losses (loss at each layer)
            contrastive_hdim (int): hidden dimension of the contrastive loss
            contrastive_loss (bool): if True, computes contrastive loss
            contrastive_align_loss (bool): if True, computes box - token alignment loss
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.attribute_embed = nn.Linear(hidden_dim, num_attributes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.contrastive_loss = contrastive_loss
        if self.contrastive_loss:
            self.contrastive_projection_image = nn.Linear(hidden_dim, contrastive_hdim, bias=False)
            self.contrastive_projection_text = nn.Linear(
                self.transformer.text_encoder.config.hidden_size, contrastive_hdim, bias=False
            )
        self.contrastive_align_loss = contrastive_align_loss
        if self.contrastive_align_loss:
            self.contrastive_align_projection_image = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_text = nn.Linear(hidden_dim, contrastive_hdim)
    
    def forward(self, samples: NestedTensor, captions, encode_and_save=True, memory_cache=None):
        """The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
                - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
                - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                            (center_x, center_y, height, width). These values are normalized in [0, 1],
                            relative to the size of each individual image (disregarding possible padding).
                            See PostProcess for information on how to retrieve the unnormalized bounding box.
                            Shape= [batch_size x num_queries x 4]
                - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """ 
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)
        
        if encode_and_save:
            assert memory_cache is None
            features, pos = self.backbone(samples)
            src, mask = features[-1].decompose()
            query_embed = self.query_embed.weight
            memory_cache = self.transformer(
                self.input_proj(src),
                mask,
                query_embed,
                pos[-1],
                captions,
                encode_and_save=True,
                text_memory=None,
                img_memory=None,
                text_attention_mask=None
            )
        
            if self.contrastive_loss:
                memory_cache["text_pooled_op"] = self.contrastive_projection_text(memory_cache["text_pooled_op"])
                memory_cache["img_pooled_op"] = self.contrastive_projection_image(memory_cache["img_pooled_op"])
            
            return memory_cache
        else:
            assert memory_cache is not None
            hs = self.transformer(
                mask=memory_cache["mask"],
                query_embed=memory_cache["query_embed"],
                pos_embed=memory_cache["pos_embed"],
                encode_and_save=False,
                text_memory=memory_cache["text_memory_resized"],
                img_memory=memory_cache["img_memory"],
                text_attention_mask=memory_cache["text_attention_mask"],
            )
            out = {}
            outputs_class = self.class_embed(hs)
            outputs_coord = self.bbox_embed(hs).sigmoid()
            outputs_attr = self.attribute_embed(hs)
            out.update(
                {
                    "pred_logits": outputs_class[-1],
                    "pred_boxes": outputs_coord[-1],
                    "pred_attributes": outputs_attr[-1],
                }
            )
            proj_queries, proj_tokens = None, None
            if self.contrastive_align_loss:
                proj_queries = F.normalize(self.contrastive_align_projection_image(hs), p=2, dim=-1)
                proj_tokens = F.normalize(
                    self.contrastive_align_projection_text(memory_cache["text_memory"]).transpose(0, 1), p=2, dim=-1
                )
                out.update(
                    {
                        "proj_queries": proj_queries[-1],
                        "proj_tokens": proj_tokens,
                        "tokenized": memory_cache["tokenized"],
                    }
                )
            if self.aux_loss:
                if self.contrastive_align_loss:
                    assert proj_tokens is not None and proj_queries is not None
                    out["aux_outputs"] = [
                        {
                            "pred_logits": a,
                            "pred_boxes": b,
                            "pred_attributes": c,
                            "proj_queries": d,
                            "proj_tokens": proj_tokens,
                            "tokenized": memory_cache["tokenized"],
                        }
                        for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], outputs_attr[:-1], proj_queries[:-1])
                    ]
                else:
                    out["aux_outputs"] = [
                        {
                            "pred_logits": a,
                            "pred_boxes": b,
                            "pred_attributes": c,
                        }
                        for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_attr[:-1])
                    ]
            return out

def build(args):
    num_classes = 255
    num_attributes = 620
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = ADETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_attributes=num_attributes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        contrastive_hdim=args.contrastive_loss_hdim,
        contrastive_loss=args.contrastive_loss,
        contrastive_align_loss=args.contrastive_align_loss
    )


    return model, None, None, None