# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
RelationFormer model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
# from torchvision.ops import nms
import matplotlib.pyplot as plt
import math
import copy

from .deformable_detr_backbone import build_backbone
from .deformable_detr_2D import build_deforamble_transformer
from .deformable_transformer_dino import DeformableTransformer

from .utils import nested_tensor_from_tensor_list, NestedTensor, inverse_sigmoid
from .dn_components import prepare_for_cdn, dn_post_process


class RelationEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # if config.MODEL.DECODER.RLN_TOKEN > 0:
        #     self.relation_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM*3, config.MODEL.DECODER.HIDDEN_DIM, 2, 3)
        # else:
        self.relation_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM*2, config.MODEL.DECODER.HIDDEN_DIM, 2, 3)
    def forward(self, x):
        x = self.relation_embed(x)
        return x


class RelationFormerDINO(nn.Module):
    """ This is the RelationFormer module that performs object detection """

    def __init__(self, 
                 encoder, 
                 decoder,
                 config):
        super().__init__()
        self.encoder = encoder
        self.transformer = decoder
        self.config = config

        self.num_queries = config.MODEL.DECODER.OBJ_TOKEN + config.MODEL.DECODER.RLN_TOKEN + config.MODEL.DECODER.DUMMY_TOKEN
        self.obj_token = config.MODEL.DECODER.OBJ_TOKEN
        self.hidden_dim = config.MODEL.DECODER.HIDDEN_DIM

        self.num_feature_levels = config.MODEL.DECODER.NUM_FEATURE_LEVELS
        self.two_stage = config.MODEL.DECODER.TWO_STAGE
        self.aux_loss = config.MODEL.DECODER.AUX_LOSS
        self.with_box_refine = config.MODEL.DECODER.WITH_BOX_REFINE
        self.num_classes = config.MODEL.NUM_CLASSES

        if config.MODEL.DECODER.RLN_TOKEN > 0:
            self.relation_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM*3, config.MODEL.DECODER.HIDDEN_DIM, 2, 3)
            self.relation_embed_dim = config.MODEL.DECODER.HIDDEN_DIM*3
        else:
            # self.relation_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM*3, config.MODEL.DECODER.HIDDEN_DIM, 2, 3)
            # self.relation_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM*2, config.MODEL.DECODER.HIDDEN_DIM, 2, 3)
            self.relation_embed = None

        #  I always use self.two_stage == True
        # if (self.two_stage == True) and (self.config.MODEL.DECODER.TWO_STAGE_TYPE == 'dino'):
        self.label_enc = nn.Embedding(self.num_classes + 1, self.hidden_dim)
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        # elif self.two_stage:
        #     self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim*2)    # why *2
        # else:
        #     self.query_embed = nn.Embedding(config.MODEL.DECODER.RLN_TOKEN, self.hidden_dim*2)

        if self.num_feature_levels > 1:
            num_backbone_outs = len(self.encoder.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.encoder.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.encoder.num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )])

        ############################
        ### Next block is from DINO
        ############################

        self.box_pred_damping = box_pred_damping = None
        self.iter_update = True
        self.dn_number = 100

        # prepare pred layers
        self.dec_pred_class_embed_share = True
        self.dec_pred_bbox_embed_share = True
        # prepare class & box embed
        _class_embed = nn.Linear(config.MODEL.DECODER.HIDDEN_DIM, 2)
        _bbox_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM, config.MODEL.DECODER.HIDDEN_DIM, 4, 3)

        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if self.dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(self.transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(self.transformer.num_decoder_layers)]
        if self.dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(self.transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(self.transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        self.relation_embed = None

        # two stage
        self.two_stage_type = 'standard'
        self.two_stage_add_query_num = 0
        two_stage_bbox_embed_share = False
        two_stage_class_embed_share = False
        dec_pred_class_embed_share = True
        dec_pred_bbox_embed_share = True
        two_stage_add_query_num = 0

        if self.two_stage_type != 'no':
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
    
            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)
    
            self.refpoint_embed = None
            if self.two_stage_add_query_num > 0:
                self.init_ref_points(two_stage_add_query_num)

        self.decoder_sa_type = 'sa'
        assert self.decoder_sa_type in ['sa', 'ca_label', 'ca_content']
        if self.decoder_sa_type == 'ca_label':
            self.label_embedding = nn.Embedding(2, config.MODEL.DECODER.HIDDEN_DIM)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        else:
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
            self.label_embedding = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, samples, seg=True, targets=None):

        if not seg and not isinstance(samples, NestedTensor):
            # When inferencing
            samples = nested_tensor_from_tensor_list(samples)
        elif seg:
            # When training
            samples = nested_tensor_from_tensor_list([tensor.expand(3, -1, -1).contiguous() for tensor in samples])

        # These below code are exactly same as Deformable DETR's
        features, pos = self.encoder(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose() 
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.encoder[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)


        # if self.config.MODEL.DECODER.TWO_STAGE_TYPE == 'dino':
            # dn_args = (targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale)
            # Always using CDN
            # default self.dn_number=100
            # When inferencing, input_query_bbox = input_query_label = attn_mask = dn_meta = None
            """
            input_query_label: [B, \max(nodes) * 2, D]
            input_query_bbox: [B, \max(nodes) * 2, 4]
            attn_mask: [\max(nodes)+num_queries, \max(nodes)+num_queries]
            이로인해 Batch 별로 shape of input_query_bbox는 바뀔 수 있다.
        """
        input_query_label, input_query_bbox, attn_mask, dn_meta =\
            prepare_for_cdn(dn_args=(targets, 100, 0.4, 0.5),
                            training=seg, num_queries=self.config.MODEL.DECODER.OBJ_TOKEN, num_classes=self.num_classes,
                            hidden_dim=self.hidden_dim, label_enc=self.label_enc)
        # else:
            # input_query_label, input_query_bbox, attn_mask, dn_meta = None

        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(srcs, masks, input_query_bbox, pos, input_query_label,attn_mask)
        # In case num object=0
        hs[0] += self.label_enc.weight[0,0]*0.0

        # deformable-detr-like anchor update
        # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig  + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)        

        outputs_class = torch.stack([layer_cls_embed(layer_hs) for
                                     layer_cls_embed, layer_hs in zip(self.class_embed, hs)])
        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord_list = \
                dn_post_process(outputs_class, outputs_coord_list,
                                dn_meta,self.aux_loss,self._set_aux_loss)
        out = {'pred_logits': outputs_class[-1], 'pred_nodes': outputs_coord_list[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)

        # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            out['enc_outputs'] = {'pred_logits': interm_class, 'pred_nodes': interm_coord}
            out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_nodes': init_box_proposal}

            # prepare enc outputs
            if hs_enc.shape[0] > 1:
                enc_outputs_coord = []
                enc_outputs_class = []
                for layer_id, (layer_box_embed, layer_class_embed, layer_hs_enc, layer_ref_enc) in enumerate(zip(self.enc_bbox_embed, self.enc_class_embed, hs_enc[:-1], ref_enc[:-1])):
                    layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                    layer_enc_outputs_coord_unsig = layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                    layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()

                    layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                    enc_outputs_coord.append(layer_enc_outputs_coord)
                    enc_outputs_class.append(layer_enc_outputs_class)

                out['enc_outputs'] = [
                    {'pred_logits': a, 'pred_nodes': b} for a, b in zip(enc_outputs_class, enc_outputs_coord)
                ]

        out['dn_meta'] = dn_meta

        return hs, out, srcs
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_nodes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def _build_mlp(self, input_dim, hidden_layers, output_dim):
        layers = [nn.Linear(input_dim, hidden_layers[0]), nn.ReLU()]
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        return nn.Sequential(*layers)

    def sample_descriptors(self, feature_map, vertex_positions):
        B, N, _ = vertex_positions.size()
        _, D, H, W = feature_map.size()

        feature_map = feature_map # [B, D, H, W]
        # vertex_positions: [0,1)
        init_vertex_positions = vertex_positions * torch.tensor([[H -1, W - 1]], device=feature_map.device)
        
        # Normalize vertex positions to [-1, 1] for grid_sample)
        vertex_positions_normalized = init_vertex_positions.float() / torch.tensor([[H - 1, W - 1]], device=feature_map.device) * 2 - 1

        # Create all pairs of vertices (B, N, N, 2)
        start_positions = vertex_positions_normalized.unsqueeze(2).unsqueeze(-1)  # Shape: [B, N, 1, 2, 1]
        end_positions = vertex_positions_normalized.unsqueeze(1).unsqueeze(-1)  # Shape: [B, 1, N, 2, 1]

        # Compute linearly spaced points between each pair of vertices
        # For simplicity, let's sample 2 points on the line between vertex pairs
        steps = torch.linspace(0, 1, steps=self.num_samples, device=feature_map.device).view(1, 1, 1, 1, self.num_samples)  # Shape: [1, 1, 1, 1, num_samples]

        # # Linear combination to find the interpolated positions
        interpolated_positions = (1 - steps) * start_positions + steps * end_positions  # Shape: [B, N, N, 2,num_samples]

        # Reshape interpolated positions for grid_sample
        # Flatten N, N, and num_samples dimensions
        interpolated_positions = interpolated_positions.view(B, N*N*self.num_samples, 1, 2)

        # Sample descriptors using grid_sample
        sampled_descriptors = F.grid_sample(feature_map, interpolated_positions, mode='bilinear', align_corners=True)

        # Reshape the sampled descriptors to match the MLP input
        # B, D, (N*N*num_samples), 1 -> B, N*N, num_samples * D
        sampled_descriptors = sampled_descriptors.squeeze(3)  # Remove the last dimension
        sampled_descriptors = sampled_descriptors.view(B, D, N*N, self.num_samples)
        sampled_descriptors = sampled_descriptors.permute(0, 2, 1, 3).reshape(B, N*N, D*self.num_samples)

        # Mask to select unique edge pairs (upper triangle excluding diagonal)
        mask = torch.triu(torch.ones(N, N, device=feature_map.device), diagonal=1).bool()
        mask_flat = mask.view(-1)  # Flatten mask for indexing

        # Apply MLP only on unique pairs
        sampled_descriptors_unique = self.mlp_edge(sampled_descriptors[:, mask_flat])

        # Create a complete N x N matrix for all edge descriptors
        edge_descriptors = torch.zeros(B, N*N, sampled_descriptors_unique.size(-1), device=feature_map.device)
        edge_descriptors[:, mask_flat] = sampled_descriptors_unique

        # Copy descriptors to reverse edges
        reverse_mask_flat = mask.t().contiguous().view(-1)
        edge_descriptors[:, reverse_mask_flat] = sampled_descriptors_unique

        # Reshape to [B, N, N, ...]
        edge_descriptors = edge_descriptors.view(B, N, N, -1) # [B, N, N, D]

        return edge_descriptors


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_relationformer_dino(config, **kwargs):

    encoder = build_backbone(config)
    decoder = build_deformable_transformer_dino(config)

    model = RelationFormerDINO(
        encoder,
        decoder,
        config,
        **kwargs
    )

    return model


def build_relation_embed(config):
    model = RelationEmbed(config)
    return model


def build_deformable_transformer_dino(config):
    """
        The Following hard-coded parameters are from .
        https://github.com/IDEA-Research/DINO/blob/main/config/DINO/DINO_4scale.py
    """
    return DeformableTransformer(
        d_model=config.MODEL.DECODER.HIDDEN_DIM,
        dropout=config.MODEL.DECODER.DROPOUT,
        nhead=config.MODEL.DECODER.NHEADS,
        num_queries=config.MODEL.DECODER.OBJ_TOKEN,
        dim_feedforward=config.MODEL.DECODER.DIM_FEEDFORWARD,
        num_encoder_layers=config.MODEL.DECODER.ENC_LAYERS,
        num_unicoder_layers=0,
        num_decoder_layers=config.MODEL.DECODER.DEC_LAYERS,
        normalize_before=False,
        return_intermediate_dec=True,
        query_dim=4, # [x, y, w, h]
        activation='relu',
        num_patterns=0,
        modulate_hw_attn=True,

        deformable_encoder=True,
        deformable_decoder=True,
        num_feature_levels=config.MODEL.DECODER.NUM_FEATURE_LEVELS,
        enc_n_points=config.MODEL.DECODER.ENC_N_POINTS,
        dec_n_points=config.MODEL.DECODER.DEC_N_POINTS,
        use_deformable_box_attn=False,
        box_attn_type='roi_align',

        learnable_tgt_init=True,
        decoder_query_perturber=None,

        add_channel_attention=False,
        add_pos_value=False,
        random_refpoints_xy=False,

        # two stage
        two_stage_type='standard',  # ['no', 'standard', 'early']
        two_stage_pat_embed=0,
        two_stage_add_query_num=0,
        two_stage_learn_wh=False,
        two_stage_keep_all_tokens=False,
        dec_layer_number=None,
        rm_self_attn_layers=None,
        key_aware_type=None,
        layer_share_type=None,

        rm_detach=None,
        decoder_sa_type='sa',
        module_seq=['sa', 'ca', 'ffn'],

        embed_init_tgt=True,
        use_detached_boxes_dec_out=False
    )