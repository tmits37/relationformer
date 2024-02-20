import torch
import torch.nn.functional as F

from .deformable_detr_backbone import build_backbone
from .relationDINO import RelationFormerDINO, build_deformable_transformer_dino

import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix

from .gnn.lanegnn import LaneGNN
from .matcher import build_matcher

from torch_geometric.data import Data, Batch


def distance_from_ref(point):
    return np.sqrt(point[:, 0]**2 + point[:, 1]**2)


def compare_distances(point_pairs):
    # Calculate distances for all first and second points in pairs
    distances_first = distance_from_ref(point_pairs[:, 0, :])
    distances_second = distance_from_ref(point_pairs[:, 1, :])
    # Compare distances and return boolean array
    return distances_first <= distances_second


def directed_graph_adj(adj, pred_node):
    # adj: [N, N]: torch.tesnr
    # pred_node: [N, 2]: torch.tensr
    device = pred_node.device
    index = np.array(np.nonzero(adj)).transpose(1,0)
    pred_node = pred_node.detach().cpu().numpy()

    result = compare_distances(pred_node[index])
    new_index = index[result]

    new_adj = np.zeros(adj.shape)
    new_adj[new_index[:, 0], new_index[:, 1]] = 1    
    return new_adj


def filter_by_distance(A, K=20):
    A_csr = A.tocsr()
    A_binary = A_csr.copy()
    A_binary.data = np.where(A_binary.data <= K, 1, 0)

    # Convert the binary distances to a permutation matrix
    A_permutation = csr_matrix((A_binary.data, A_binary.indices, A_binary.indptr), shape=A_binary.shape)
    return A_permutation


def pad_graph(node_features, adj_matrix, N):
    num_nodes = node_features.shape[0]

    padding = N - num_nodes
    padded_features = F.pad(node_features, (0, 0, 0, padding), mode='constant', value=0)
    padded_adj = F.pad(adj_matrix, (0, padding, 0, padding), mode='constant', value=0)
    return padded_features, padded_adj


def get_adj_matrix(pred_nodes, num_link=3, distance=50, include_self=False, directed=False):
    device = pred_nodes[0].device
    edges = []
    adjs = []
    for b in pred_nodes:
        # number of the neighborhoods should be equal or less than the number of samples
        b_num_link = min(num_link, b.shape[0] - 1)
        if b_num_link <= 0:
            edges.append([])
            adjs.append(torch.zeros((1,1), device=device))
        else:
            A = kneighbors_graph(b.detach().cpu().numpy(), b_num_link, mode='distance', include_self=include_self)
            A_permutation = filter_by_distance(A, K=distance).toarray()
            if directed:
                A_permutation = directed_graph_adj(A_permutation, b)
            loc = np.array(np.where(A_permutation)) # [2, K]
            loc = loc.transpose(1,0) # [K, 2]
            edges.append(torch.tensor(loc, device=device))
            adjs.append(torch.tensor(A_permutation, device=device))
    return edges, adjs


def pad_nodes_adjs(node_features, adjs, N=128):
    device = node_features[0].device

    pns = []
    pas = []
    for n, a in zip(node_features, adjs):
        pn, pa = pad_graph(n, a, N)
        pns.append(pn)
        pas.append(pa)
    pns = torch.stack(pns, dim=0).to(device)
    pas = torch.stack(pas, dim=0).to(device)

    return pns, pas


class RelationFormerDINOGNN(RelationFormerDINO):
    def __init__(self, encoder,  decoder, config):
        super(RelationFormerDINOGNN, self).__init__(encoder, decoder, config)

        self.H = self.config.DATA.IMG_SIZE[0] 
        self.edge_descriptors = config.MODEL.EDGE_DESCRIPTORS
        if self.edge_descriptors:
            print("you are now using edge descriptors")
            self.num_samples = 2
            self.relation_embed = None
        else:
            self.relation_embed = None

        self.obj_token = config.MODEL.DECODER.OBJ_TOKEN
        self.rln_token = config.MODEL.DECODER.RLN_TOKEN

        if config.MODEL.DECODER.APPEND_LOC:
            graph_learner_input_dim = config.MODEL.DECODER.HIDDEN_DIM + 2
        else:
            graph_learner_input_dim = config.MODEL.DECODER.HIDDEN_DIM

        self.append_loc = config.MODEL.DECODER.APPEND_LOC
        self.graph_learner = LaneGNN(gnn_depth=6, 
                                    edge_geo_dim=16, 
                                    map_feat_dim=64, 
                                    edge_dim=32, 
                                    node_dim=16, 
                                    msg_dim=32, 
                                    map_token_dim=int(config.MODEL.DECODER.HIDDEN_DIM * self.num_samples))
        self.matcher = build_matcher(config)
        self.directed = config.MODEL.GNN.DIRECTED

    def forward(self, samples, seg=True, targets=None):
        hs, out, srcs = super(RelationFormerDINOGNN, self).forward(samples, seg=seg, targets=targets)

        # get the valid points
        h = hs[-1]
        object_token = h[...,:self.obj_token,:]

        # Generally, the number of valid token is larger than target_nodes:
        if targets:
            cvt_node = [k['nodes'] for k in targets]
            cvt_edge = [k['edges'] for k in targets]
            cvts = {'nodes':cvt_node, 'edges':cvt_edge}
            indices = self.matcher(out, cvts)
            valid_token = [i for i, j in indices]
        else:
            valid_token = torch.argmax(out['pred_logits'], -1).detach()

        # indices = self.matcher(out, target)
        # ind_pred = [i for i, j in indices] # 578 this is same number with original 

        # import pickle
        # with open('/nas/k8s/dev/research/doyoungi/valid_token_matcher.pkl', 'wb') as fp:
        #     pickle.dump(valid_token_matcher, fp)

        device = h.device
        node_features = []
        pred_nodes = []
        for batch_id in range(h.shape[0]):
            if targets:
                node_id = valid_token[batch_id].to(device)
            else:
                node_id = torch.nonzero(valid_token[batch_id]).squeeze(1)

            pred_nodes.append(out['pred_nodes'][batch_id, node_id, :2].detach())
            if self.append_loc:
                node_feature = torch.cat([object_token[batch_id, node_id, :], out['pred_nodes'][batch_id, node_id, :2]], dim=-1)
            else:
                node_feature = object_token[batch_id, node_id, :]
            node_features.append(node_feature)

        # Pred_nodes = [B, K, 2]
        # pred_nodes = out['pred_nodes'][..., :2]
        edges, adjs = get_adj_matrix(pred_nodes,  num_link=6, distance=40/128, include_self=True, directed=self.directed)
        pad_node_features, pad_adj = pad_nodes_adjs(node_features, adjs)
        pad_adj = pad_adj.float()

        # torch.save(valid_token, '/nas/k8s/dev/research/doyoungi/valid_token.pt')
        # torch.save(pad_node_features, '/nas/k8s/dev/research/doyoungi/pad_node_features.pt')
        # torch.save(pad_adj, '/nas/k8s/dev/research/doyoungi/pad_adj.pt')
        # torch.save(out['pred_nodes'].detach(), '/nas/k8s/dev/research/doyoungi/pred_nodes.pt')

        pred_nodes = out['pred_nodes'][..., :2] # [B, N, 2]
        edge_descriptors = self.sample_descriptors(srcs[0], pred_nodes)
        out['edge_descriptors'] = edge_descriptors
        # torch.save(out['edge_descriptors'].detach(), '/nas/k8s/dev/research/doyoungi/edge_descriptors.pt')
        
        batch_graph, batch_edge_index = self.convert_to_torch_geometric_graph(pred_nodes, 
                                                       pad_adj, 
                                                       valid_token, 
                                                       edge_descriptors,
                                                       targets=targets)
        # [E, 1], [N, 1], [N, 1]
        # edge_classifier, node_classifier, endpoint_classifer = self.graph_learner(batch_graph)
        edge_classifier, endpoint_classifer = self.graph_learner(batch_graph)

        out['edge_classifier'] = edge_classifier
        # out['node_classifier'] = node_classifier
        out['endpoint_classifier'] = endpoint_classifer
        out['edge_index'] = batch_edge_index

        # [B, 1, 1024]
        # h = h + torch.mm(self.mlp_edge(torch.zeros((1, 1024), device=h.device)), 
        #                        torch.zeros((256,1), device=h.device))

        # z_node_features = self.graph_learner(pad_node_features, pad_adj)
        # # If entered GNN, it's shape shuld be same as pad_node_features
        # for batch_id in range(h.shape[0]):
        #     node_id = torch.nonzero(valid_token[batch_id]).squeeze(1)
        #     # non_zero_node_feature_id = torch.nonzero(torch.sum(pad_node_features[batch_id], dim=1)).squeeze(1)
        #     h[batch_id, node_id, :] = z_node_features[batch_id, :len(node_id)]
        # nearest-K-uppder bounds permutation matrix

        # import pickle
        # with open('/nas/k8s/dev/research/doyoungi/out.pkl', 'wb') as f:
        #     pickle.dump(out, f)

        return h, out, srcs

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
        # sampled_descriptors_unique = self.mlp_edge(sampled_descriptors[:, mask_flat])
        sampled_descriptors_unique = sampled_descriptors[:, mask_flat]

        # Create a complete N x N matrix for all edge descriptors
        edge_descriptors = torch.zeros(B, N*N, sampled_descriptors_unique.size(-1), device=feature_map.device)
        edge_descriptors[:, mask_flat] = sampled_descriptors_unique

        # Copy descriptors to reverse edges
        reverse_mask_flat = mask.t().contiguous().view(-1)
        edge_descriptors[:, reverse_mask_flat] = sampled_descriptors_unique

        # Reshape to [B, N, N, ...]
        edge_descriptors = edge_descriptors.view(B, N, N, -1) # [B, N, N, D]

        return edge_descriptors

    def convert_to_torch_geometric_graph(self,
                                         pred_nodes, 
                                        pad_adj, 
                                        valid_token, 
                                        edge_descriptors,
                                        targets=None):

        device = pred_nodes.device
        batch_num = pred_nodes.size(0)
        # valid_token = valid_token.detach().cpu().numpy()

        batch_node_feat = []
        batch_edge_attr = []
        batch_edge_img_feat = []
        batch_edge_index = []
        for batch_id in range(batch_num):
            if targets:
                node_id = valid_token[batch_id].to(device)
            else:
                node_id = torch.nonzero(valid_token[batch_id]).squeeze(1)

            # node_id = np.nonzero(valid_token[batch_id])[0]
            # node_id = valid_token_matcher[batch_id]
            # non_zero_node_feature_id = torch.nonzero(torch.sum(pad_node_features[batch_id], dim=1)).squeeze(1)
            pred_node = pred_nodes[batch_id, node_id][:,:2].detach().cpu().numpy()
            adj = pad_adj[batch_id, :len(node_id), :len(node_id)]

            node_feats_list = []
            node_gt_list = []
            for i, anchor in enumerate(pred_node):
                node_tensor = torch.tensor([anchor[0], anchor[1]], device=device).reshape(1, -1)
                node_feats_list.append(node_tensor)

            edge_img_feats = []
            edge_attr_list = []
            edge_index = []
            edges = np.array(np.nonzero(adj.detach().cpu().numpy())).transpose(1,0)

            if len(edges) == 0:
                batch_edge_attr.append(torch.tensor([], dtype=torch.float32, device=device).reshape(2, 0))
                batch_edge_img_feat.append(torch.tensor([], dtype=torch.float32, device=device).reshape(2, 0))
                batch_edge_index.append(torch.tensor([], dtype=torch.long, device=device).reshape(2, 0))
            else:
                for edge_idx, edge in enumerate(edges):
                    i, j = edge
                    s_x, s_y = pred_node[i][1], pred_node[i][0]
                    e_x, e_y = pred_node[j][1], pred_node[j][0]

                    # if params.preprocessing.visualize:
                    # ax.arrow(s_x, s_y, e_x-s_x, e_y-s_y, color="red", width=0.005, head_width=0.05, shape='left')

                    delta_x, delta_y = e_x - s_x, e_y - s_y
                    mid_x, mid_y = s_x + delta_x/2, s_y + delta_y/2

                    edge_len = np.sqrt(delta_x**2 + delta_y**2)
                    edge_angle = np.arctan(delta_y/(delta_x + 1e-6))

                    edge_tensor = torch.tensor([edge_angle, edge_len, mid_x, mid_y], device=device).reshape(1, -1)
                    edge_attr_list.append(edge_tensor)

                    n, m = i, j
                    if i > j:
                        n = j
                        m = i

                    edge_attr_tensor = edge_descriptors[batch_id, n, m].clone().detach()
                    edge_img_feats.append(edge_attr_tensor)
                    edge_index.append(torch.tensor([i, j], device=device))

                batch_edge_attr.append(torch.cat(edge_attr_list, dim=0))
                batch_edge_img_feat.append(torch.stack(edge_img_feats, dim=0))
                batch_edge_index.append(torch.stack(edge_index, dim=1))

            batch_node_feat.append(torch.cat(node_feats_list, dim=0))
            
        graphs = [construct_graph(batch_node_feat[i], 
                                batch_edge_attr[i], 
                                batch_edge_img_feat[i], 
                                batch_edge_index[i], 
                                batch_idx=i) for i in range(batch_num)]
        batch_graph = Batch.from_data_list(graphs)
        return batch_graph, batch_edge_index

def build_relationformer_dino_gnn(config, **kwargs):
    encoder = build_backbone(config)
    decoder = build_deformable_transformer_dino(config)

    model = RelationFormerDINOGNN(
        encoder,
        decoder,
        config,
        **kwargs
    )
    
    return model


def construct_graph(node_feat, edge_attr, edge_img_feat, edge_index, batch_idx): 
    graph_data = Data(x=node_feat, 
                    edge_img_feats=edge_img_feat,
                    edge_attr=edge_attr,
                    edge_index=edge_index,
                    batch_idx=batch_idx)
    return graph_data

