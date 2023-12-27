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
from .utils import nested_tensor_from_tensor_list, NestedTensor, inverse_sigmoid

from .fcn_head import NestedFCNHead


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


class RelationFormer(nn.Module):
    """ This is the RelationFormer module that performs object detection """

    def __init__(self, 
                 encoder, 
                 decoder,
                 config):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

        self.num_queries = config.MODEL.DECODER.OBJ_TOKEN + config.MODEL.DECODER.RLN_TOKEN + config.MODEL.DECODER.DUMMY_TOKEN
        self.obj_token = config.MODEL.DECODER.OBJ_TOKEN
        self.hidden_dim = config.MODEL.DECODER.HIDDEN_DIM

        self.num_feature_levels = config.MODEL.DECODER.NUM_FEATURE_LEVELS
        self.two_stage = config.MODEL.DECODER.TWO_STAGE
        self.aux_loss = config.MODEL.DECODER.AUX_LOSS
        self.with_box_refine = config.MODEL.DECODER.WITH_BOX_REFINE
        self.num_classes = config.MODEL.NUM_CLASSES

        # input_dim, hidden_dim, output_dim, num_layers
        self.class_embed = nn.Linear(config.MODEL.DECODER.HIDDEN_DIM, 2)
        self.bbox_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM, config.MODEL.DECODER.HIDDEN_DIM, 4, 3)
        
        if config.MODEL.DECODER.RLN_TOKEN > 0:
            self.relation_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM*3, config.MODEL.DECODER.HIDDEN_DIM, 2, 3)
            self.relation_embed_dim = config.MODEL.DECODER.HIDDEN_DIM*3
        else:
            # self.relation_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM*3, config.MODEL.DECODER.HIDDEN_DIM, 2, 3)
            # self.relation_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM*2, config.MODEL.DECODER.HIDDEN_DIM, 2, 3)
            self.relation_embed = None

        if not self.two_stage:
            self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim*2)    # why *2
        else:
            self.query_embed = nn.Embedding(config.MODEL.DECODER.RLN_TOKEN, self.hidden_dim*2)

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

        # This block exist in Deformable-DETR, but not in Relationformer
        # prior_prob = 0.01
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        # nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        # nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        # for proj in self.input_proj:
        #     nn.init.xavier_uniform_(proj[0].weight, gain=1)
        #     nn.init.constant_(proj[0].bias, 0)


        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (decoder.decoder.num_layers + 1) if self.two_stage else decoder.decoder.num_layers
        # The previous code block
        # if self.two_stage:
        #     self.class_embed = _get_clones(self.class_embed, num_pred)
        #     self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
        #     nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
        #     # hack implementation for iterative bounding box refinement
        #     self.decoder.decoder.bbox_embed = self.bbox_embed

        #     self.decoder.decoder.class_embed = self.class_embed
        #     for box_embed in self.bbox_embed:
        #         nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        # self.decoder.decoder.bbox_embed = None # This is default code, not with_box_refine
        if self.with_box_refine:
            print('with_box_refine')
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.decoder.decoder.bbox_embed = self.bbox_embed
        else:
            print('not with_box_refine')
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            # self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            # self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.decoder.decoder.bbox_embed = None
        if self.two_stage:
            print('two_stage')
            # hack implementation for two-stage
            self.decoder.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        self.seg = config.MODEL.SEG
        if self.seg:
            self.aux_fpn_head = NestedFCNHead(origin_shape=config.DATA.IMG_SIZE)

        self.edge_descriptors = config.MODEL.EDGE_DESCRIPTORS
        if self.edge_descriptors:
            print("you are now using edge descriptors")
            self.num_samples = 4
            self.mlp_edge = self._build_mlp(
                input_dim=int(config.MODEL.DECODER.HIDDEN_DIM * self.num_samples),  # Concatenation of descriptors from two vertices
                hidden_layers=[config.MODEL.DECODER.HIDDEN_DIM*1],
                output_dim=config.MODEL.DECODER.HIDDEN_DIM
            )


    def forward(self, samples, seg=True):

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

        # self.num_feature_levels = 4, len(srcs) = 3
        # Generally, self.num_feature_levles = len(srcs) = 4
        # This is the hyperparmeter of the encoder; Conv + Downsampling
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

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        else:
            query_embeds = self.query_embed.weight # -> Relaiton Token
    
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.decoder(
            srcs, masks, query_embeds, pos
        )

        # For including the gradient propagation in this forward function
        # hs = hs + 0 * relation_embed
        if self.relation_embed:
            hs = hs + torch.mm(self.relation_embed(torch.zeros((1, self.relation_embed_dim), device=hs.device)), torch.zeros((2,1), device=hs.device))

        if self.two_stage:
            outputs_classes = []
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                outputs_class = self.class_embed[lvl](hs[lvl][:,:self.obj_token,:])
                tmp = self.bbox_embed[lvl](hs[lvl][:,:self.obj_token,:])
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)
            class_prob = torch.stack(outputs_classes)
            coord_loc = torch.stack(outputs_coords)

            out = {'pred_logits': class_prob[-1], 'pred_nodes': coord_loc[-1]}
            hs = hs[-1] #  Do I need this lines?

        else:
            # below code are different from Deformable DETR's
            object_token = hs[...,:self.obj_token,:]
            class_prob = self.class_embed(object_token) # same from DETR's, Reliability of the obj. [B, OBJ_TOKENS, 2]
            coord_loc = self.bbox_embed(object_token).sigmoid() # same from DETR's, [B, OBJ_TOKENS, 4]

            out = {'pred_logits': class_prob, 'pred_nodes': coord_loc}

        if self.edge_descriptors:
            edge_descriptors = self.sample_descriptors(features[0].decompose()[0], coord_loc[:,:,:2])

        # Auxiliary Head
        # Normalliy, hs, class_prob, coord_loc: [B, OBJ_TOKENS+1, D], [B, OBJ_TOKENS, 2], [B, OBJ_TOKENS, 4]
        if self.seg:
            # torch.Size([256, 512, 16, 16]) torch.Size([256, 1024, 8, 8]) torch.Size([256, 2048, 4, 4])
            seg_logits = self.aux_fpn_head(features)
            out['pred_segs'] = seg_logits

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_nodes': enc_outputs_coord}

            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(class_prob, coord_loc)

        if self.edge_descriptors:
            out['edge_descriptors'] = edge_descriptors

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


def build_relationformer(config, **kwargs):

    encoder = build_backbone(config)
    decoder = build_deforamble_transformer(config)

    model = RelationFormer(
        encoder,
        decoder,
        config,
        **kwargs
    )

    return model


def build_relation_embed(config):
    model = RelationEmbed(config)
    return model