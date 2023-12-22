from abc import ABC
import math

import torch
from torch import nn


class StreetMoverDistance(ABC):
    r"""[summary]

    Args:
        ABC ([type]): [description]

    Raises:
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]

    Returns:
        [type]: [description]
    """    ''''''
    def __init__(self, eps, max_iter, reduction="mean"):
        """[summary]

        Args:
            eps ([type]): [description]
            max_iter ([type]): [description]
            reduction (str, optional): [description]. Defaults to "mean".
        """        ''''''
        super(StreetMoverDistance, self).__init__()
        self.reduction = reduction
        self.sinkhorn_distance = SinkhornDistance(eps=eps, max_iter=max_iter, reduction=reduction)
        # self.buffer_num: int = 0
        # self._buffers: Optional[List[List[torch.Tensor]]] = None
        # self._synced_tensors: Optional[List[Optional[torch.Tensor]]] = None
        # self._synced: bool = False

    def __call__(self, node_list, edge_list, pred_node_list, pred_edge_list):  # type: ignore
        """[summary]

        Args:
            node_list ([type]): [description]
            edge_list ([type]): [description]
            pred_node_list ([type]): [description]
            pred_edge_list ([type]): [description]

        Returns:
            [type]: [description]
        """        ''''''
        # node_list = [torch.cat((node, torch.zeros((node.shape[0], 1)).cuda()), dim=1) for node in node_list]
        # pred_node_list = [torch.cat((node, torch.zeros((node.shape[0], 1)).cuda()), dim=1) for node in pred_node_list]
        ret = self._compute_list(node_list, edge_list, pred_node_list, pred_edge_list)

        return ret

    def _compute_list(self, node_list, edge_list, pred_node_list, pred_edge_list):
        """[summary]

        Args:
            node_list ([type]): [description]
            edge_list ([type]): [description]
            pred_node_list ([type]): [description]
            pred_edge_list ([type]): [description]

        Returns:
            [type]: [description]
        """        ''''''
        ret=[]
        # compute dice (BxC) for each channel for each batch
        for nodes, edges, pred_nodes, pred_edges in zip(node_list, edge_list, pred_node_list, pred_edge_list):
            # print(nodes.shape, edges.shape)
            A = torch.zeros((nodes.shape[0], nodes.shape[0]))
            A[edges[:,0],edges[:,1]] = 1

            pred_A = torch.zeros((pred_nodes.shape[0], pred_nodes.shape[0]))
            if nodes.shape[0]>1 and pred_nodes.shape[0]>1 and pred_edges.size != 0:
                # print(pred_edges)
                pred_A[pred_edges[:,0], pred_edges[:,1]] = 1.0

                ret.append(compute_meanSMD(
                    A.T, nodes, pred_A.T, pred_nodes, self.sinkhorn_distance, n_points=100
                ))
            else:
                ret.append(torch.tensor([1], dtype=torch.float)) # TODO: fix the loss mismatch issue

        ret = torch.cat(ret, dim=0)
        return ret



def compute_meanSMD(y_A, y_nodes, output_A, output_nodes, sinkhorn_distance, n_points=100):
    """[summary]

    Args:
        y_A ([type]): [description]
        y_nodes ([type]): [description]
        output_A ([type]): [description]
        output_nodes ([type]): [description]
        sinkhorn_distance ([type]): [description]
        n_points (int, optional): [description]. Defaults to 100.

    Returns:
        [type]: [description]
    """    ''''''
    y_pc = get_point_cloud(y_A, y_nodes, n_points)
    output_pc = get_point_cloud(output_A, output_nodes, n_points)

    # from matplotlib import pyplot as plt
    # for elem in [y_pc, output_pc]:
    #     x, y = elem.T
    #     plt.scatter(x,y)
    #     plt.show()

    sink_dist, P, C = sinkhorn_distance(y_pc, output_pc)
    return sink_dist #(y_pc, output_pc), (sink_dist, P, C)


def get_point_cloud(A, nodes, n_points):
    n_divisions = n_points - 1 + 0.01
    total_len = get_cumulative_distance(A, nodes)
    step = total_len / n_divisions
    points = []
    next_step = 0.
    used_len = 0.

    for i in range(A.shape[0]):
        for j in range(i):
            if A[i, j] == 1.:
                next_step, used, pts = get_points(next_step, step, nodes[j].clone(), nodes[i].clone())
                used_len += used
                points += pts
                last_node = nodes[i].clone()
                # plot_point_cloud(adj[0], coord[0], pts)
    # trick in case we miss points, due to approximations in python computation of distances
    if 0 < len(points) < n_points:
        while len(points) < n_points:
            points.append((last_node[0].item(), last_node[1].item()))
    # if the graph has no edges, create point cloud with 100 points in (0,0)
    if len(points) == 0:
        return torch.zeros((100, 2))
        # print(f"The point cloud has an expected number of points: {len(points)} instead of {n_points}")
    # print(f"Generated {len(points)} points using {used_len}/{total_len} length")
    return torch.FloatTensor(points)


def get_cumulative_distance(A, nodes):
    tot = 0.
    for i in range(A.shape[0]):
        for j in range(i):
            # print(i, j)
            if A[i, j] == 1.:
                # print(nodes[i], nodes[j])
                tot += euclidean_distance(nodes[i], nodes[j])
    return tot


def get_points(next_step, step, a, b):
    l = euclidean_distance(a, b)
    m = ((b[1] - a[1]) / (b[0] - a[0])).item()
    sign_x = -1 if b[0] < a[0] else 1  # going backwards or forward
    sign_y = -1 if b[1] < a[1] else 1  # going backwards or forward
    pts = []
    used = 0
    while next_step < l:
        used += next_step
        l -= next_step
        dx = sign_x * next_step / math.sqrt(1 + m ** 2)
        dy = m * dx if abs(dx) > 1e-06 else sign_y * next_step
        a[0] += dx
        a[1] += dy
        pts.append((a[0].item(), a[1].item()))
        next_step = step
    next_step = step - l
    return next_step, used, pts


def euclidean_distance(a, b):
    return math.sqrt((a - b).pow(2).sum().item())


# From https://github.com/dfdazac/wassdistance
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
    
    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]
        
        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()
        
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1
        
        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()
            
            actual_nits += 1
            if err.item() < thresh:
                break
        
        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))
        
        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()
        return torch.tensor([cost]), pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps
    
    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C
    
    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1
