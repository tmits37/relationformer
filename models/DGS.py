# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Modules to compute the matching cost and solve the corresponding LSAP."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph
from torch import nn

from models.matcher_sinkhorn import Sinkhorn


def filter_by_distance(A, K=20):
    A_csr = A.tocsr()
    A_binary = A_csr.copy()
    A_binary.data = np.where(A_binary.data <= K, 1, 0)

    # Convert the binary distances to a permutation matrix
    A_permutation = csr_matrix(
        (A_binary.data, A_binary.indices, A_binary.indptr),
        shape=A_binary.shape)
    return A_permutation


def get_adj_matrix(pred_nodes, num_link=3, distance=50, include_self=False):
    device = pred_nodes[0].device
    edges = []
    adjs = []
    for b in pred_nodes:
        b_num_link = min(num_link, b.shape[0] - 1)
        if b_num_link <= 0:
            edges.append([])
            adjs.append(torch.zeros((1, 1), device=device))
        else:
            A = kneighbors_graph(b.detach().cpu().numpy(),
                                 b_num_link,
                                 mode='distance',
                                 include_self=include_self)
            A_permutation = filter_by_distance(A, K=distance)
            loc = np.array(np.where(A_permutation.toarray()))  # [2, K]
            loc = loc.transpose(1, 0)  # [K, 2]
            edges.append(torch.tensor(loc, device=device))
            adjs.append(torch.tensor(A_permutation.toarray(), device=device))
    return edges, adjs


def edge_list_to_adj_matrix(edge_list):
    vertices = np.unique(edge_list.flatten())
    vertex_to_index = {vertex: i for i, vertex in enumerate(vertices)}

    num_vertices = len(vertices)
    adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)

    for edge in edge_list:
        src, dest = edge
        adj_matrix[vertex_to_index[src], vertex_to_index[dest]] = 1

    return adj_matrix


def remove_vertices_and_redirect_edges(adj_matrix, vertices_to_remove):
    n = adj_matrix.shape[0]
    sorted_vertices_to_remove = sorted(vertices_to_remove, reverse=True)

    adj_matrix = adj_matrix.copy()
    for vertex_to_remove in sorted_vertices_to_remove:
        if vertex_to_remove < 0 or vertex_to_remove >= n:
            print(vertex_to_remove, n)
            raise ValueError('Vertex index out of bounds')

        incoming = np.where(adj_matrix[:, vertex_to_remove] == 1)[0]
        outgoing = np.where(adj_matrix[vertex_to_remove, :] == 1)[0]

        if len(incoming) > 0 and len(outgoing) > 0:
            for i in incoming:
                for j in outgoing:
                    adj_matrix[i, j] = 1

        adj_matrix[vertex_to_remove, :] = 0  # Set entire row to 0
        adj_matrix[:, vertex_to_remove] = 0  # Set entire column to 0

        adj_matrix[vertex_to_remove,
                   vertex_to_remove] = 1  # Set entire row to 0

    return adj_matrix


def generate_adj_mask(adj_matrix,
                      pred_nodes,
                      n=256,
                      k=170,
                      mask_type='K-N',
                      weight_mask=0.2):
    mask = np.ones((n, n))
    if mask_type == 'K-N':
        mask[k:, :] = 0
        mask[:, k:] = 0
    elif mask_type == 'diag':
        diag = np.diag(np.ones(n)) == 1
        mask[diag] = weight_mask
    elif mask_type == 'nearest':
        _, adjs = get_adj_matrix(pred_nodes,
                                 num_link=5,
                                 distance=40 / 320,
                                 include_self=True)
        mask = adj_matrix + adjs.detach().numpy()
        mask[mask == 2] = 1  # union operation

        # randomly convert from 0 to 1
        zero_indices = np.where(mask == 0)
        zero_positions = list(zip(zero_indices[0], zero_indices[1]))
        num_zeros_to_change = min(int(256 * 256 * 0.005), len(zero_positions))
        selected_positions = np.random.choice(range(len(zero_positions)),
                                              size=num_zeros_to_change,
                                              replace=False)
        for pos in selected_positions:
            mask[zero_positions[pos]] = 1
    else:
        raise AssertionError
    return mask


def sinkhorn_matcher(C, num_queries, sizes, device):
    indices = []
    pred_marginal = torch.ones(1, num_queries).to(device)
    for i, c in enumerate(C.split(sizes, -1)):
        cost_mat = c[i].unsqueeze(0)
        gt_marginal = torch.ones(1, sizes[i]).to(device)
        cost_mat_transposed = cost_mat.transpose(1, 2).to(device)
        result = Sinkhorn.apply(cost_mat_transposed, gt_marginal,
                                pred_marginal, 100, 1e-2)

        dims = result.size()
        pred_idx = np.array([], dtype=int)
        gt_idx = np.array([], dtype=int)

        result_np = (result.detach().cpu().numpy().squeeze(0))

        pred_idx = np.empty(0, dtype=int)
        gt_idx = np.empty(0, dtype=int)

        for row in range(dims[1]):
            max_proba_cols = np.where(
                result_np[row] == np.max(result_np[row]))[0]
            pred_idx = np.append(pred_idx, max_proba_cols)
            gt_idx = np.append(gt_idx,
                               np.full(len(max_proba_cols), row, dtype=int))

            sorted_indices = np.argsort(pred_idx)
            sorted_pred_idx = pred_idx[sorted_indices]
            sorted_gt_idx = gt_idx[sorted_indices]

        indices.append((sorted_pred_idx, sorted_gt_idx))
    return indices


def nearest_match_nodes(pred_nodes, gt_nodes, min_dist=12):
    """Matches each ground truth node with the nearest predicted node.

    Parameters:
    gt_nodes (torch.Tensor): Ground truth nodes of shape (N, 2),
                             where N is the number of nodes.
    pred_nodes (torch.Tensor): Predicted nodes of shape (M, 2),
                            where M is the number of nodes.
    Returns:
    List[Tuple]: A list of tuples where each tuple contains the indices of
                 the matched gt and pred node.
    """
    matched_pairs = []
    # Compute pairwise distances between gt and predicted nodes
    distances = torch.cdist(gt_nodes, pred_nodes, p=1)  # Shape: (N, M)
    for i in range(gt_nodes.size(0)):
        # Find the nearest predicted node for each gt node
        distances_to_gt = distances[i]
        min_distance, min_index = torch.min(distances_to_gt, dim=0)
        if min_distance < min_dist:
            matched_pairs.append((min_index.item(), i))
            distances[:, min_index] = float('inf')
        else:
            pass

    matched_pairs = sorted(matched_pairs)
    matched_pairs = np.array(matched_pairs)

    return [matched_pairs[:, 0], matched_pairs[:, 1]]


def wrapping_nearest_match_nodes(pred_nodes, gt_nodes, width=320, min_dist=12):
    bs = pred_nodes.size(0)
    indices = []
    for b in range(bs):
        pred_node = pred_nodes[b].clone() * width
        gt_node = gt_nodes[b].clone() * width
        index = nearest_match_nodes(pred_node.float(),
                                    gt_node.float(),
                                    min_dist=min_dist)
        indices.append(index)
    return indices


class Matcher(nn.Module):
    """This class computes an assignment between the targets and the
    predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of
    this, in general, there are more predictions than targets. In this case, we
    do a 1-to-1 matching of the best predictions, while the others are un-
    matched (and thus treated as non-objects).
    """
    def __init__(self, config=None, add_logit=False, matcher='Hungarian'):
        super().__init__()
        self.add_logit = add_logit
        self.cost_nodes = 5
        self.cost_class = 3
        self.matcher = matcher
        if config:
            self.mask_type = config.MODEL.ADJ_MAT_MASK
            self.weight_mask = config.MODEL.W_MASK
        else:
            self.mask_type = 'None'
            self.weight_mask = 1

        if config.MODEL.DATATYPE == 'road':
            self.road_adj = True
        else:
            self.road_adj = False

    @torch.no_grad()  # 역전파 안 한다는 뜻
    def forward(self, outputs, targets):
        # outputs = {'pred_logits':..., 'pred_nodes':tensor(32, 128, 4)}
        bs, num_queries = outputs['pred_nodes'].shape[:2]
        out_nodes = outputs['pred_nodes'][...].flatten(0, 1)

        tgt_nodes = torch.cat([v for v in targets['nodes']])
        cost_nodes = torch.cdist(out_nodes, tgt_nodes,
                                 p=1)  # 4096, 722, L1 로스값 텐서

        heatmaps = outputs['pred_heatmaps'].sigmoid()
        width = heatmaps.size(2)  # heatmaps must have same width and height
        for b in range(bs):
            b_desc = heatmaps[b, 0]
            b_graph = outputs['pred_nodes'][b] * width
            b_graph = b_graph.long()

            b_desc = b_desc[b_graph[:, 0], b_graph[:, 1]]

            if b == 0:
                sel_desc = b_desc.unsqueeze(0)
            else:
                sel_desc = torch.cat((sel_desc, b_desc.unsqueeze(0)), dim=0)

        tgt_ids = torch.cat([
            torch.tensor([1] * v.shape[0]).to(out_nodes.device)
            for v in targets['nodes']
        ])  # [1]*551
        sel_desc = torch.stack([sel_desc, sel_desc],
                               dim=2).to(out_nodes.device)
        cost_class = -sel_desc.flatten(0, 1)[..., tgt_ids]

        C = (self.cost_nodes * cost_nodes + self.cost_class * cost_class
             )  # 4096,722 + 4096,722

        C = C.view(bs, num_queries, -1).cpu()  # 16,256,1049

        sizes = [len(v) for v in targets['nodes']]
        if self.matcher == 'Hungarian':
            indices = [
                linear_sum_assignment(c[i])
                for i, c in enumerate(C.split(sizes, -1))
            ]
        elif self.matcher == 'Sinkhorn':
            indices = sinkhorn_matcher(C,
                                       num_queries,
                                       sizes,
                                       device=out_nodes.device)
        elif self.matcher == 'Nearest':
            try:
                indices = wrapping_nearest_match_nodes(outputs['pred_nodes'],
                                                       targets['nodes'],
                                                       width=width,
                                                       min_dist=12)
            except:  # noqa
                indices = [
                    linear_sum_assignment(c[i])
                    for i, c in enumerate(C.split(sizes, -1))
                ]

        results, masks = self.generate_directed_adjacency_matrix(
            outputs, targets, indices)
        results = torch.tensor(results, device=out_nodes.device)
        masks = torch.tensor(masks, device=out_nodes.device)
        return results, masks, indices

    def generate_directed_adjacency_matrix(self, outputs, targets, indices):
        bs = len(indices)
        adj_matrixes = []
        mask_matrixes = []
        n = outputs['pred_nodes'].size(1)
        for b in range(bs):
            mapping = {
                j: i
                for i, j in zip(indices[b][0].tolist(), indices[b][1].tolist())
            }

            edges = targets['edges'][b].tolist()
            k = len(targets['nodes'][b].tolist())

            non_matched_key = [
                i for i in range(k) if i not in list(mapping.keys())
            ]
            adj_ori = edge_list_to_adj_matrix(np.array(edges))
            adj_modi = remove_vertices_and_redirect_edges(
                adj_ori, non_matched_key)
            mod_edges = np.array(np.where(adj_modi == 1)).transpose(
                1, 0)  # gt_sequence

            adj_matrix = np.zeros((n, n))
            for e in mod_edges:
                if not set(e) & set(non_matched_key):
                    adj_matrix[mapping[e[0]], mapping[e[1]]] = 1

            # end rule
            not_in_index = [x for x in range(n) if x not in mapping.values()]
            for e in not_in_index:
                adj_matrix[e, e] = 1

            # end_rule for road dataset
            if self.road_adj:
                # print('end_rule for road_dataset')
                uniq_element, counts = np.unique(mod_edges.flatten(),
                                                 return_counts=True)
                element_once = uniq_element[counts == 1]
                end_node = mod_edges[:, 1]
                element_once_end_node = [
                    x for x in element_once if x in end_node
                ]
                for e in element_once_end_node:
                    adj_matrix[mapping[e], mapping[e]] = 1

            # Masking-rule
            mask = generate_adj_mask(
                adj_matrix=adj_matrix,
                pred_nodes=outputs['pred_nodes'][b].unsqueeze(0),
                n=n,
                k=k,
                mask_type=self.mask_type,
                weight_mask=self.weight_mask)
            adj_matrixes.append(adj_matrix)
            mask_matrixes.append(mask)

        adj_matrixes = np.stack(adj_matrixes, axis=0)
        mask_matrixes = np.stack(mask_matrixes, axis=0)
        return adj_matrixes, mask_matrixes
