from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from .loss import SetCriterion

def find_terminal_nodes(node_array, edge_array):
    node_counter = Counter()
    for edge in edge_array.flatten():
        node_counter[edge] += 1

    terminal_nodes = [node for node, count in node_counter.items() if count == 1]

    whether_terminal_node = np.zeros(node_array.shape[0]).astype('int64')
    whether_terminal_node[terminal_nodes] = 1
    return whether_terminal_node


class GraphCriterion(SetCriterion):
    def __init__(self, config, matcher, net, distributed):
        super(GraphCriterion, self).__init__(config, matcher, net, distributed)
        self.directed = config.MODEL.GNN.DIRECTED
        print(self.directed)

    def loss_edges(self, out, target, indices):
        batch_edge_index = out['edge_index']
        edge_classifier = out['edge_classifier']
        target_edges = target['edges']

        target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in zip(target_edges, indices)]
        target_edges = [torch.stack(t, 0) if len(t)>0 else torch.zeros((0,2), dtype=torch.long).to(h.device) for t in target_edges]
        new_target_edges = []
        for t, (_, i) in zip(target_edges, indices):
            tx = t.clone().detach()
            for idx, k in enumerate(i):
                t[tx==k]=idx
            new_target_edges.append(t)

        weights = []
        labels = []
        for batch_id in range(len(target['edges'])):
            target_edge = new_target_edges[batch_id]
            # directed target edge to undirected

            if not self.directed:
                target_edge = torch.cat([target_edge, torch.flip(target_edge, dims=[1])], dim=0)
            edge_index = batch_edge_index[batch_id].transpose(1, 0) # pred 

            # Expand A and B for broadcasting
            A_expanded = edge_index.unsqueeze(1)  # Shape [K, 1, 2]
            B_expanded = target_edge.unsqueeze(0)  # Shape [1, N, 2]
            matches = A_expanded == B_expanded
            row_matches = matches.all(dim=2)
            element_in_B = row_matches.any(dim=1).float()

            # weight
            weight = torch.ones(element_in_B.shape, device=edge_classifier.device)
            self_connect = edge_index[:,0] == edge_index[:,1]
            weight[self_connect] = 0

            labels.append(element_in_B)
            weights.append(weight)

        labels = torch.cat(labels)
        weights = torch.cat(weights)

        meaningful_edge_candidates = torch.sum(weights)
        loss = F.binary_cross_entropy_with_logits(edge_classifier.squeeze(1), labels, weights, reduction='sum')
        loss = loss / meaningful_edge_candidates

        return loss
    
    def loss_class(self, out, target, indices):
        endpoint_classifier = out['endpoint_classifier'] # [N, 1]

        endpoints = []
        for batch_id in range(len(target['edges'])):
            target_node = target['nodes'][batch_id] # [K,2]
            target_edge = target['edges'][batch_id]

            endpoint = find_terminal_nodes(target_node.detach().cpu().numpy(),
                                            target_edge.detach().cpu().numpy())
            endpoint = torch.tensor(endpoint, device=target_node.device)
            endpoints.append(endpoint)

        target_endpoints = torch.cat([t[i] for t, (_, i) in zip(endpoints, indices)], dim=0)
        loss = F.binary_cross_entropy_with_logits(endpoint_classifier.squeeze(1), target_endpoints.float(), reduction='mean')
        return loss

    def forward(self, h, out, target):
        indices = self.matcher(out, target)

        losses = {}
        losses['edges'] = self.loss_edges(out, target, indices)
        losses['class'] = self.loss_class(out, target, indices)

        losses_keys = list(losses.keys())
        losses['total'] = sum([losses[key] for key in losses_keys])
        return losses



    