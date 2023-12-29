import torch
import torch.nn.functional as F
from torch import nn

from models.backbone_R2U_Net import R2U_Net_origin, NonMaxSuppression, DetectionBranch
from models.DiG_generator import DiG_generator


class TopDiG(nn.Module): # TODO 클래스 만들기
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
        # self.cls_token = cls_token
        # self.hidden_dim = config.MODEL.DECODER.HIDDEN_DIM
        self.device = device
        self.v = None
        # self.init = True
        # self.use_proj_in_dec = use_proj_in_dec
        
        # self.input_proj = nn.Conv3d(encoder.num_features, config.MODEL.DECODER.HIDDEN_DIM, kernel_size=1)
        
        # self.class_embed = nn.Linear(config.MODEL.DECODER.HIDDEN_DIM, 2)
        # self.coord_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM, config.MODEL.DECODER.HIDDEN_DIM, 6, 3)
        # self.relation_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM*3, config.MODEL.DECODER.HIDDEN_DIM, 2, 3)

    # TODO Descriptor extraction 구현하기
    def descriptor_extraction(feat, ver_pos):
        B = len(feat)

        for b in range(B): # 배치 사이즈만큼 반복하여 전체 디스크립터(sel_desc) 만들기
            b_desc = feat[b]
            b_graph = ver_pos[b]

            # Extract descriptors 디스크립터 뽑기
            b_desc = b_desc[:, b_graph[:,0], b_graph[:,1]]

            # Concatenate descriptors in batches
            if b == 0:                    
                sel_desc = b_desc.unsqueeze(0)
            else:
                sel_desc = torch.cat((sel_desc, b_desc.unsqueeze(0)), dim=0) # 최종적으로 B, D, N (B, 64,256)
        return sel_desc

    def forward(self, samples):
        # R2U-Net -> DiG generator
        feature_map = self.encoder(samples)
        vertices_detection_mask = self.detectionBranch(feature_map)
        vertices_positions = self.nms(vertices_detection_mask)
        self.v = vertices_positions
        # visual_descriptor = self.descriptor_extraction(feature_map, vertices_positions) # TODO
        visual_descriptor = torch.randn(8, 256, 64).to(self.device)
        predicted_adjacency_matrix = self.decoder(vertices_positions[1], visual_descriptor)

        return predicted_adjacency_matrix


def build_TopDiG(config, **kwargs):
    encoder = R2U_Net_origin()
    nms = NonMaxSuppression()
    decoder = DiG_generator()
    detectionBranch = DetectionBranch()
    

    model = TopDiG(
        encoder,
        decoder,
        nms,
        detectionBranch,
        config,
        **kwargs
    )
    return model