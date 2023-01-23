# copyright (c) Subramanya N. Licensed under the Apache License 2.0. All Rights Reserved
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .transformer import build_transformer
from .matcher import build_matcher
from utils import box_ops
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
            # print(samples)
            features, pos = self.backbone(samples)
            # print("features", features)
            src, mask = features[-1].decompose()
            # print("src", src)
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
            # print("src", memory_cache)
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

class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, eos_coef, losses, temperature):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        self.temperature = temperature
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, positive_map, indices, num_boxes, positive_att=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """

        logits = outputs["pred_logits"].log_softmax(-1)  # BS x (num_queries) x (num_tokens)

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)

        tgt_pos = positive_map[tgt_idx]
        target_sim = torch.zeros_like(logits)
        target_sim[:, :, -1] = 1
        target_sim[src_idx] = tgt_pos

        loss_ce = -(logits * target_sim).sum(-1)

        eos_coef = torch.full(loss_ce.shape, self.eos_coef, device=target_sim.device)
        eos_coef[src_idx] = 1

        loss_ce = loss_ce * eos_coef
        loss_ce = loss_ce.sum() / num_boxes

        losses = {"loss_ce": loss_ce}

        return losses

    def loss_attributes(self, outputs, targets, positive_map, indices, num_boxes, positive_att=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """

        logits = outputs["pred_attributes"].log_softmax(-1)  # BS x (num_queries) x (num_tokens)
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)

        tgt_pos = positive_att[tgt_idx]
        target_sim = torch.zeros_like(logits)
        target_sim[:, :, -1] = 1
        target_sim[src_idx] = tgt_pos

        loss_ce = -(logits * target_sim).sum(-1)
        eos_coef = torch.full(loss_ce.shape, self.eos_coef, device=target_sim.device)
        eos_coef[src_idx] = 1

        loss_ce = loss_ce * eos_coef
        loss_ce = loss_ce.sum() / num_boxes

        losses = {"loss_attributes": loss_ce}
        return losses

    def loss_contrastive_align(self, outputs, targets, positive_map, indices, num_boxes, positive_att=None):
        bs = outputs["proj_queries"].shape[0]
        tokenized = outputs["tokenized"]

        normalized_text_emb = outputs["proj_tokens"]  # BS x (num_tokens) x hdim
        normalized_img_emb = outputs["proj_queries"]  # BS x (num_queries) x hdim

        logits = (
            torch.matmul(normalized_img_emb, normalized_text_emb.transpose(-1, -2)) / self.temperature
        )  # BS x (num_queries) x (num_tokens)

        # construct a map such that positive_map[k, i,j] = True iff query i is associated to token j in batch item k
        # For efficency, the construction happens on CPU, then the whole matrix is transferred to GPU in one go.
        positive_map = torch.zeros(logits.shape, dtype=torch.bool)
        for i, ((idx_src, idx_tgt), tgt) in enumerate(zip(indices, targets)):
            if "tokens_positive" in tgt:
                cur_tokens = [tgt["tokens_positive"][j] for j in idx_tgt]
            else:
                cur_tokens = [tgt["tokens"][j] for j in idx_tgt]

            for j, tok_list in enumerate(cur_tokens):
                for (beg, end) in tok_list:
                    beg_pos = tokenized.char_to_token(i, beg)
                    end_pos = tokenized.char_to_token(i, end - 1)
                    if beg_pos is None:
                        try:
                            beg_pos = tokenized.char_to_token(beg + 1)
                            if beg_pos is None:
                                beg_pos = tokenized.char_to_token(beg + 2)
                        except:
                            beg_pos = None
                    if end_pos is None:
                        try:
                            end_pos = tokenized.char_to_token(end - 2)
                            if end_pos is None:
                                end_pos = tokenized.char_to_token(end - 3)
                        except:
                            end_pos = None
                    if beg_pos is None or end_pos is None:
                        continue

                    assert beg_pos is not None and end_pos is not None
                    positive_map[i, idx_src[j], beg_pos : end_pos + 1].fill_(True)

        positive_map = positive_map.to(logits.device)
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits  # .masked_fill(positive_map, -1000000)

        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        neg_term = negative_logits.logsumexp(2)

        nb_pos = positive_map.sum(2) + 1e-6

        box_to_token_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~boxes_with_pos, 0).sum()

        tokens_with_pos = positive_map.any(1)
        pos_term = positive_logits.sum(1)
        neg_term = negative_logits.logsumexp(1)

        nb_pos = positive_map.sum(1) + 1e-6

        tokens_to_boxes_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~tokens_with_pos, 0).sum()
        tot_loss = (box_to_token_loss + tokens_to_boxes_loss) / 2

        return {"loss_contrastive_align": tot_loss / num_boxes}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, positive_map, indices, num_boxes, positive_att=None):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        ## Count the number of predictions that are NOT "no-object" (which is the last class)
        # normalized_text_emb = outputs["proj_tokens"]  # BS x (num_tokens) x hdim
        # normalized_img_emb = outputs["proj_queries"]  # BS x (num_queries) x hdim

        # logits = torch.matmul(
        #    normalized_img_emb, normalized_text_emb.transpose(-1, -2)
        # )  # BS x (num_queries) x (num_tokens)
        # card_pred = (logits[:, :, 0] > 0.5).sum(1)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, positive_map, indices, num_boxes, positive_att=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, positive_map, indices, num_boxes, positive_att=None):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = NestedTensor.from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, positive_map, indices, num_boxes, positive_att=None, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
            "contrastive_align": self.loss_contrastive_align,
            "attributes": self.loss_attributes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, positive_map, indices, num_boxes, positive_att, **kwargs)

    def forward(self, outputs, targets, positive_map, positive_att):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, positive_map)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / torch.distributed.get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, positive_map, indices, num_boxes, positive_att))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, positive_map)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, positive_map, indices, num_boxes, positive_att, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

class ContrastiveCriterion(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, pooled_text, pooled_image):

        normalized_text_emb = F.normalize(pooled_text, p=2, dim=1)
        normalized_img_emb = F.normalize(pooled_image, p=2, dim=1)

        logits = torch.mm(normalized_img_emb, normalized_text_emb.t()) / self.temperature
        labels = torch.arange(logits.size(0)).to(pooled_image.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        loss = (loss_i + loss_t) / 2.0
        return loss

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

    matcher = build_matcher(args)
    weight_dict = {"loss_ce": args.ce_loss_coef, "loss_bbox": args.bbox_loss_coef, "loss_attributes": args.attributes_loss_coef}
    if args.contrastive_loss:
        weight_dict["contrastive_loss"] = args.contrastive_loss_coef
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    weight_dict["loss_giou"] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "cardinality", "attributes"]
    if args.masks:
        losses += ["masks"]
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]
    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        eos_coef=args.eos_coef,
        losses=losses,
        temperature=args.temperature_NCE,
    )
    contrastive_criterion = None
    if args.contrastive_loss:
        contrastive_criterion = ContrastiveCriterion(temperature=args.temperature_NCE)

    return model, criterion, contrastive_criterion, weight_dict