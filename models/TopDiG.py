import torch
import torch.nn.functional as F
from torch import nn

from models.backbone_R2U_Net import R2U_Net_origin, NonMaxSuppression, DetectionBranch
from models.DiG_generator import DiG_generator
from models.optimal_matching import OptimalMatching


class TopDiG(nn.Module):
    """ This is the RelationFormer module that performs object detection """

    def __init__(self, encoder, decoder, nms, detectionBranch, config, device='cuda'):
        """[summary]

        Args:
            encoder ([type]): [description]
            decoder ([type]): [description]
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.nms = nms
        self.detectionBranch = detectionBranch
        self.device = device
        self.v = None # 피쳐 맵
        self.h = None # 히트 맵

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
        feature_map = self.encoder(samples)
        vertices_detection_mask = self.detectionBranch(feature_map)
        self.h = vertices_detection_mask
        vertices_positions = self.nms(vertices_detection_mask)
        self.v = vertices_positions
        visual_descriptor = self.descriptor_extraction(feature_map, vertices_positions[1]) # B, D, 256
        predicted_adjacency_matrix = self.decoder(vertices_positions[1], visual_descriptor)
        # predicted_adjacency_matrix = self.decoder(feature_map, vertices_positions[1]) # TODO 폴리월드 디코더

        return predicted_adjacency_matrix


def build_TopDiG(config, **kwargs):
    # load the checkpoint of backbone
    # pretrained_backbone = torch.load(config.BACKBONE_PATH, map_location='cpu')
    encoder = R2U_Net_origin()
    detectionBranch = DetectionBranch()
    # encoder.load_state_dict(pretrained_backbone['model_state_dict'], strict=False) # TODO 디버깅 해보기
    # detectionBranch.load_state_dict(pretrained_backbone['model_state_dict'], strict=False) # TODO 디버깅 해보기
    nms = NonMaxSuppression()
    decoder = DiG_generator()
    # decoder = OptimalMatching() # TODO 월요일에 백본 로드해도 학습 안 될 경우 갈아끼우기
    
    model = TopDiG(
        encoder,
        decoder,
        nms,
        detectionBranch,
        config,
        **kwargs
    )
    return model