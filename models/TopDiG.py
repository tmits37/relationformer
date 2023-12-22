import torch
import torch.nn.functional as F
from torch import nn

from backbone_R2U_Net import R2U_Net_origin, NonMaxSuppression, DetectionBranch
from DiG_generator import DiG_generator


class TopDiG(nn.Module): # TODO 클래스 만들기
    """ This is the RelationFormer module that performs object detection """

    def __init__(self, encoder, decoder, filter, detectionBranch, config, device='cuda'):
        """[summary]

        Args:
            encoder ([type]): [description]
            decoder ([type]): [description]
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.filter = filter
        self.detectionBranch = detectionBranch
        # self.cls_token = cls_token
        # self.hidden_dim = config.MODEL.DECODER.HIDDEN_DIM
        self.device = device
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
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        
        # swin transformer
        # feat_list, mask_list, pos_list = self.encoder(samples, self.position_embedding, return_interm_layers=False)
        
        # seresnet
        # feat_list = [self.encoder(samples)]
        # mask_list = [torch.zeros(feat_list[0][:, 0, ...].shape, dtype=torch.bool).to(feat_list[0].device)]
        # pos_list = [self.position_embedding(mask_list[-1]).to(feat_list[0].device)]
        
        # query_embed = self.query_embed.weight
        # h = self.decoder(self.input_proj(feat_list[-1]), mask_list[-1], query_embed, pos_list[-1])
        # object_token = h[...,:-1,:]

        # class_prob = self.class_embed(object_token)
        # coord_loc = self.coord_embed(object_token).sigmoid()
        
        # out = {'pred_logits': class_prob, 'pred_nodes': coord_loc}

        # R2U-Net -> DiG generator
        feature_map = self.encoder(samples)
        vertices_detection_mask = self.detectionBranch(feature_map)
        vertices_positions = self.filter(vertices_detection_mask)
        visual_descriptor = self.descriptor_extraction(feature_map, vertices_positions)
        predicted_adjacency_matrix = self.decoder(feature_map, visual_descriptor)

        return predicted_adjacency_matrix


def build_TopDiG(config, **kwargs):
    encoder = R2U_Net_origin()
    filter = NonMaxSuppression()
    decoder = DiG_generator()
    detectionBranch = DetectionBranch()
    

    model = TopDiG(
        encoder,
        decoder,
        filter,
        detectionBranch,
        config,
        **kwargs
    )
    return model