import torch
import torch.nn.functional as F
from torch import nn

from .deformable_detr_backbone import build_backbone
from .relationDINO import RelationFormerDINO, build_deformable_transformer_dino, MLP
# from utils import scores_to_permutations, permutations_to_polygons


class RelationFormerDINOPOLY(RelationFormerDINO):
    def __init__(self, encoder,  decoder, config):
        super(RelationFormerDINOPOLY, self).__init__(encoder, decoder, config)

        self.H = self.config.DATA.IMG_SIZE[0] 
        self.edge_descriptors = config.MODEL.EDGE_DESCRIPTORS
        if self.edge_descriptors:
            print("you are now using edge descriptors")
            self.num_samples = 4
            self.mlp_edge = self._build_mlp(
                input_dim=int(config.MODEL.DECODER.HIDDEN_DIM * self.num_samples),  # Concatenation of descriptors from two vertices
                hidden_layers=[config.MODEL.DECODER.HIDDEN_DIM*1],
                output_dim=config.MODEL.DECODER.HIDDEN_DIM
            )
            self.relation_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM*3, config.MODEL.DECODER.HIDDEN_DIM, 2, 3)
            self.relation_embed_dim = config.MODEL.DECODER.HIDDEN_DIM*3
        else:
            self.relation_embed = None

        self.pyramid_descriptor = config.MODEL.PYRAMID_DESCRIPTORS
        if self.pyramid_descriptor:
            self.mlp_edge_forward = self._build_mlp(
                input_dim=int(config.MODEL.DECODER.HIDDEN_DIM),  # Concatenation of descriptors from two vertices
                hidden_layers=[config.MODEL.DECODER.HIDDEN_DIM*1],
                output_dim=config.MODEL.DECODER.HIDDEN_DIM
            )

    def forward(self, samples, seg=True, targets=None):
        hs, out, srcs = super(RelationFormerDINOPOLY, self).forward(samples, seg=seg, targets=targets)

        # Features srcs: len(srcs) = 4
        # srcs[-1].shape = [B, D, H/64, W/64]
        # srcs[0].shape = [B, D, H/8, W/8]
        # Descriptor hs: len(hs) = 4
        # hs[-1].shape = [B, 324, D]
        pred_nodes = out['pred_nodes'][..., :2] # [B, N, 2]
        # pad_loc = out['dn_meta']['pad_size'] if out['dn_meta'] else 0
        # sel_desc = hs[-1][:, pad_loc:, :] # [B, N, D]
        # [B, N, D] -> [B, N', D] Actural nodes?
        # sel_desc, offset = self.gnn(sel_desc, pred_nodes)
        # refined_pred_nodes = pred_nodes + offset * self.correction_radius

        if self.edge_descriptors:
            if self.pyramid_descriptor:
                for i, src in enumerate(srcs[::-1]): # len(srcs) = 4
                    if i == 0:
                        edge_descriptors = self.sample_descriptors(src, pred_nodes) # [B, N, N, D], e0
                    else:
                        edge_descriptors = self.update_sample_descriptors(edge_descriptors, src, pred_nodes)
            else:
                edge_descriptors = self.sample_descriptors(srcs[0], pred_nodes)
            
        out['edge_descriptors'] = edge_descriptors

        hs = hs[-1]
        if self.relation_embed:
            hs = hs + torch.mm(self.relation_embed(torch.zeros((1, self.relation_embed_dim), device=hs.device)), 
                               torch.zeros((2,1), device=hs.device))
        #  pred_logits = out['pred_logits']
        return hs, out, srcs
    
    def update_sample_descriptors(self, edge_descriptors, feature_map, vertex_positions):
        B, N, _ = vertex_positions.size()
        _, D, H, W = feature_map.size()

        b_edge_descriptors = super(RelationFormerDINOPOLY, self).sample_descriptors(feature_map, vertex_positions)
        b_edge_descriptors = b_edge_descriptors.view(B, N*N, D)

        mask = torch.triu(torch.ones(N, N, device=feature_map.device), diagonal=1).bool()
        mask_flat = mask.view(-1)  # Flatten mask for indexing
        
        edge_descriptors = edge_descriptors.view(B, N*N, D)
        edge_descriptors = self.mlp_edge_forward(edge_descriptors[:, mask_flat])
        edge_descriptors = edge_descriptors + b_edge_descriptors[:, mask_flat]

        new_edge_descriptors = torch.zeros(B, N*N, edge_descriptors.size(-1), device=feature_map.device)
        new_edge_descriptors[:, mask_flat] = edge_descriptors
        new_edge_descriptors = new_edge_descriptors.view(B, N, N, -1) # [B, N, N, D]
        return new_edge_descriptors



def build_relationformer_dino_poly(config, **kwargs):
    encoder = build_backbone(config)
    decoder = build_deformable_transformer_dino(config)

    model = RelationFormerDINOPOLY(
        encoder,
        decoder,
        config,
        **kwargs
    )
    
    return model