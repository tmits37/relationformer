import torch
import torch.nn.functional as F
from torch import nn

from models.backbone_R2U_Net import build_backbone
from models.DiG_generator import DiG_generator
from models.optimal_matching import OptimalMatching


class TopDiG(nn.Module):
    def __init__(self, backbone, decoder, config, device='cuda'):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.device = device

    def descriptor_extraction(self, feature_map, vertices_positions):
        B = len(feature_map)

        for b in range(B): # 배치 사이즈만큼 반복하여 전체 디스크립터(sel_desc) 만들기
            b_desc = feature_map[b]
            b_graph = vertices_positions[b]

            # Extract descriptors 디스크립터 뽑기
            b_desc = b_desc[:, b_graph[:,0], b_graph[:,1]] # 64, N

            # 차원 변경: (64, N) -> (N, 64)
            b_desc = b_desc.transpose(0, 1)

            # Concatenate descriptors in batches
            if b == 0:
                sel_desc = b_desc.unsqueeze(0)
            else: # 최종적으로 B, D, N (B, 64, 256)
                sel_desc = torch.cat((sel_desc, b_desc.unsqueeze(0)), dim=0)
        return sel_desc

    def forward(self, samples):
        # R2U-Net -> DiG generator
        feature_map, vertices_detection_mask, vertices_positions = self.backbone(samples)
        visual_descriptor = self.descriptor_extraction(feature_map, vertices_positions[1]) # B, D, 256
        scores1, scores2 = self.decoder(vertices_positions[1], visual_descriptor)
        # scores1, scores2 = self.decoder.predict(feature_map, vertices_positions[1]) # 폴리월드 디코더
        output = {
            'pred_nodes': (vertices_positions[1]/320).to(self.device),
            'pred_heatmaps': vertices_detection_mask,
            'scores1': scores1,
            'scores2': scores2
            }

        return output


def build_TopDiG(config, **kwargs): # 일단 ptm 쓰는게 default인 상태
    pretrained_backbone = torch.load(config.DATA.BACKBONE_PATH, map_location='cpu')
    backbone = build_backbone(config.DATA.BACKBONE_CONFIG_PATH)
    backbone.load_state_dict(pretrained_backbone['model_state_dict'])
    decoder = DiG_generator()
    # decoder = OptimalMatching() # 폴리월드 디코더
    
    model = TopDiG(
        backbone,
        decoder,
        config,
        **kwargs
    )

    return model