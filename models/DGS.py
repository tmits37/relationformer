# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np

B = 8
N = 256


def generate_directed_adjacency_matrix(pairs_list, n = N):
    """ 주어진 노드 쌍 리스트를 기반으로 방향성 인접 행렬을 생성합니다. """
    adjacency_matrix = np.zeros((n, n), dtype=int)

    # 인접 행렬 채우기
    for i, j in pairs_list:
        adjacency_matrix[i, j] = 1  # 방향성 그래프: i에서 j로의 방향

    return adjacency_matrix

class HungarianMatcher(nn.Module): # relationformer의 matcher.py에서 가져옴
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, outputs, targets):
        # outputs = {'pred_logits':..., 'pred_nodes':tensor(32, 128, 4)}
        # 32개의 배치, 128차원, 4개는 bbox_embed MLP 리턴값 느낌이 앞두개 한점, 뒤두개 다른점
        # r2u의 경우 8, 256, 2가 들어온다
        bs, num_queries = outputs['pred_nodes'].shape[:2] # 8, 256

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, 2] # 4096, 2
        out_nodes = outputs['pred_nodes'][...].flatten(0, 1) # 우리는 2048, 2

        # Also concat the target labels and boxes
        tgt_nodes = torch.cat([v for v in targets['nodes']]) # 722, 2

        # Compute the L1 cost between nodes
        cost_nodes = torch.cdist(out_nodes, tgt_nodes, p=1) # 4096, 722, L1 로스값 텐서

        # Compute the cls cost
        # tgt_ids = torch.cat([torch.tensor([1]*v.shape[0]).to(out_nodes.device) for v in targets['nodes']]) # [1]*551
        # v는 32개의 배치인데 하나마다 타겟 노드들 들고 있음
        # 즉 타겟 노드들은 1이라는 클래스를 주는 텐서를 만드는 과정. 값은 1만 갖고 있음 
        # cost_class = -outputs["pred_logits"].flatten(0, 1).softmax(-1)[..., tgt_ids]
        # print(cost_class[0][0])
        # 4096,2(배경이냐, 노드냐) -> 4096, 1(노드로짓) -> 4096, 722(정답 노드 수 만큼 복사하여 늘리기)
        # 코스트가 1에 가까운지 비교하기 위해 로짓값을 소프트맥스해주고 타겟 551개의 위치에 대한 코스트 클래스 완성

        # Final cost matrix
        # cost_nodes: 3, cost_class: 5
        # C = self.cost_nodes * cost_nodes + self.cost_class * cost_class # 4096,722 + 4096,722
        # 우리는 로짓을 모른다
        C = cost_nodes # 4096,722
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v) for v in targets['nodes']]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # 배치별로 나눠서 (32,128,24),(32,128,9)...
        # indices에 헝가리안 결과값 존재. 32개의 2차원 xy 리스트
        # 텐서로 담아서 리턴
        # 이거는 인퍼런스용으로 매칭 되는 것만 알면 됨
        # return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        # 로스 계산하려면 adj_mat 만들어서 리턴해야 함
        # tgt_edges = target['edges']
        # pred_edges = [[mapping[i],mapping[j]] for i, j in tgt_edges]
        return indices
        return generate_directed_adjacency_matrix(indices)


if __name__ == "__main__":
    matcher = HungarianMatcher()
    matcher.eval()
    output = { # R2U넷의 결과를 nms 돌려서 나온 결과
        "pred_nodes": torch.randn(B, N, 2), # x_cord, y_cord
        # "pred_logits": torch.randn(B, N, 2) # 배경_로짓, 노드_로짓
    }
    target = {'nodes':[
        torch.tensor([[0.0859, 0.0781],
        [0.5625, 0.2266],
        [0.7109, 0.7109],
        [0.7188, 0.2266],
        [0.0859, 0.5781],
        [0.0859, 0.2344],
        [0.7188, 0.3906],
        [0.7188, 0.0703],
        [0.7188, 0.5547],
        [0.4062, 0.2344],
        [0.8750, 0.2266],
        [0.0781, 0.9297],
        [0.0859, 0.4062],
        [0.0859, 0.7500],
        [0.7109, 0.8750],
        [0.2500, 0.2344],
        [0.1396, 0.9531],
        [0.0859, 0.0391],
        [0.0391, 0.2344],
        [0.7188, 0.0391],
        [0.7109, 0.9531],
        [0.9531, 0.2266],
        [0.0391, 0.9102]]),
        torch.tensor([[0.9375, 0.4766],
        [0.9375, 0.3281],
        [0.9375, 0.7812],
        [0.9453, 0.1719],
        [0.9375, 0.6250],
        [0.9453, 0.9297],
        [0.9453, 0.0391],
        [0.9453, 0.9531]]),
        torch.tensor([[0.1172, 0.5703],
        [0.3906, 0.6953],
        [0.9297, 0.1328],
        [0.3281, 0.8750],
        [0.4141, 0.9141],
        [0.2812, 0.8125],
        [0.4297, 0.5391],
        [0.8750, 0.1328],
        [0.8750, 0.4062],
        [0.8750, 0.2656],
        [0.3750, 0.9453],
        [0.7500, 0.4141],
        [0.2031, 0.6953],
        [0.4531, 0.8750],
        [0.5312, 0.4219],
        [0.0391, 0.4638],
        [0.9531, 0.1228],
        [0.3806, 0.9531]]),
        torch.tensor([[0.6250, 0.6875],
        [0.3672, 0.2578],
        [0.0859, 0.2578],
        [0.5078, 0.2578],
        [0.7812, 0.0938],
        [0.8672, 0.3047],
        [0.8906, 0.3516],
        [0.7891, 0.2578],
        [0.2266, 0.2578],
        [0.7969, 0.5391],
        [0.4531, 0.7031],
        [0.7891, 0.3984],
        [0.6484, 0.2578],
        [0.8516, 0.2578],
        [0.8203, 0.2578],
        [0.8047, 0.8281],
        [0.7969, 0.6797],
        [0.2891, 0.7344],
        [0.1250, 0.7734],
        [0.9531, 0.6797],
        [0.0391, 0.7930],
        [0.7788, 0.0391],
        [0.9531, 0.2578],
        [0.8047, 0.9531]]),
        torch.tensor([[0.2969, 0.2031],
        [0.3828, 0.8125],
        [0.7109, 0.8516],
        [0.7109, 0.7656],
        [0.2344, 0.8203],
        [0.9297, 0.1875],
        [0.1328, 0.2031],
        [0.7734, 0.1953],
        [0.8828, 0.8438],
        [0.5469, 0.8516],
        [0.0859, 0.8203],
        [0.5469, 0.7734],
        [0.4531, 0.2031],
        [0.6094, 0.2031],
        [0.4531, 0.0469],
        [0.8828, 0.7578],
        [0.4531, 0.0391],
        [0.0391, 0.1984],
        [0.9531, 0.1875],
        [0.9531, 0.8270],
        [0.0391, 0.8228],
        [0.9531, 0.7444]]),
        torch.tensor([[0.9531, 0.7578],
        [0.7734, 0.4688],
        [0.1562, 0.4531],
        [0.5938, 0.1797],
        [0.3047, 0.3672],
        [0.6875, 0.3281],
        [0.8594, 0.6094],
        [0.5156, 0.0547],
        [0.4453, 0.2656],
        [0.0391, 0.5312],
        [0.9531, 0.9521],
        [0.9516, 0.9531],
        [0.5064, 0.0391]]),
        torch.tensor([[0.3516, 0.8281],
        [0.6641, 0.2266],
        [0.6641, 0.3750],
        [0.6641, 0.6797],
        [0.6641, 0.0703],
        [0.0391, 0.8594],
        [0.9531, 0.0781],
        [0.6641, 0.8281],
        [0.1953, 0.8281],
        [0.6641, 0.5234],
        [0.5078, 0.8281],
        [0.6641, 0.9531],
        [0.6641, 0.0391],
        [0.8828, 0.0391]]),
        torch.tensor([[0.6875, 0.1719],
        [0.3984, 0.9141],
        [0.6016, 0.0391],
        [0.6250, 0.6875],
        [0.9375, 0.5859],
        [0.7344, 0.5703],
        [0.8516, 0.4531],
        [0.5078, 0.8047],
        [0.7656, 0.3125],
        [0.3594, 0.9531],
        [0.9531, 0.6115]])]
    }
    out = matcher(output, target)
    # print(out)
    print(out[0][0])
    print(len(out[0][0]))
    print(out[0][1])
    print(len(out[0][1]))

    print(output['pred_nodes'][0][out[0][0][0]])

