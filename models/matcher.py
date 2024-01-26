# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Modules to compute the matching cost and solve the corresponding LSAP."""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the
    predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of
    this, in general, there are more predictions than targets. In this case, we
    do a 1-to-1 matching of the best predictions, while the others are un-
    matched (and thus treated as non-objects).
    """
    def __init__(self, config):
        """Creates the matcher.

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost # noqa
        """
        super().__init__()
        self.cost_nodes = config.MODEL.MATCHER.C_NODE
        self.cost_class = config.MODEL.MATCHER.C_CLASS

    @torch.no_grad()
    def forward(self, outputs, targets):
        """[summary]

        Args:
            outputs ([type]): [description]
            targets ([type]): [description]

        Returns:
            [type]: [description]
        """
        # outputs = {'pred_logits':..., 'pred_nodes':tensor(32, 128, 4)}
        # 32개의 배치, 128차원, 4개는 bbox_embed MLP 리턴값 느낌이 앞두개 한점, 뒤두개 다른점
        bs, num_queries = outputs['pred_nodes'].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_nodes = outputs['pred_nodes'][..., :2].flatten(
            0, 1)  # [batch_size * num_queries, 2] # 4096, 2

        # Also concat the target labels and boxes
        tgt_nodes = torch.cat([v for v in targets['nodes']])  # 722, 2

        # Compute the L1 cost between nodes
        cost_nodes = torch.cdist(out_nodes, tgt_nodes, p=1)  # 4096, 722

        # Compute the cls cost
        tgt_ids = torch.cat([
            torch.tensor([1] * v.shape[0]).to(out_nodes.device)
            for v in targets['nodes']
        ])  # [1]*551
        # v는 32개의 배치인데 하나마다 타겟 노드들 들고 있음
        # 즉 타겟 노드들은 1이라는 클래스를 주는 텐서를 만드는 과정. 값은 1만 갖고 있음
        cost_class = (
            -outputs['pred_logits'].flatten(0, 1).softmax(-1)[..., tgt_ids]
        )  # 4096, 2 -> 4096, 722(코스트)
        # 코스트가 1에 가까운지 비교하기 위해 로짓값을 소프트맥스해주고 타겟 551개에 대한 코스트 클래스 완성

        # Final cost matrix
        # cost_nodes: 3, cost_class: 5
        C = (self.cost_nodes * cost_nodes + self.cost_class * cost_class
             )  # 4096,722 + 4096,722
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v) for v in targets['nodes']]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]
        # 배치별로 나눠서 (32,128,24),(32,128,9)...
        # indices에 헝가리안 결과값 존재. 32개의 2차원 xy 리스트
        # 텐서로 담아서 리턴
        return [(
            torch.as_tensor(i, dtype=torch.int64),
            torch.as_tensor(j, dtype=torch.int64),
        ) for i, j in indices]


def build_matcher(config):
    return HungarianMatcher(config)
