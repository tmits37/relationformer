import torch
from torch import nn


class SetCriterion(nn.Module):
    """This class computes the loss for Graphformer.

    The process happens in two steps:     1) we compute hungarian assignment
    between ground truth boxes and the outputs of the model     2) we supervise
    each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, config, matcher, net, distributed):
        """Create the criterion.

        Parameters:
            num_classes: number of object categories, omitting the special no-object category # noqa
            matcher: module able to compute a matching between targets and proposals # noqa
            weight_dict: dict containing as key the names of the losses and as values their relative weight. # noqa
            eos_coef: relative classification weight applied to the no-object category # noqa
            losses: list of all the losses to be applied. See get_loss for list of available losses. # noqa
        """
        super().__init__()
        self.config = config
        self.matcher = matcher
        self.net = net
        self.distributed = distributed
        self.adj_mat_mask = config.MODEL.ADJ_MAT_MASK
        self.losses = config.TRAIN.LOSSES
        self.weight_dict = {  # 이 가중치도 조정가능
            'node': config.TRAIN.W_NODE,
            'graph': config.TRAIN.W_GRAPH,
        }
        self.loss_weighted_ce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
            [100]).cuda())  # 100에서 조정가능
        self.loss_ce = nn.BCEWithLogitsLoss()

    def loss_node(self, output, target):
        """pretrain처럼 PolyWorld와 같은 weighted BCE 사용하겠습니다."""
        loss = self.loss_weighted_ce(output, target)
        return loss

    def loss_graph(self, scores1, scores2, target, masked_mat):
        """sigmoid로 scores 한칸 한칸 값의 범위를 (0~1) 스케일로 바꿔주겠다.

        TopDiG처럼 BCE 사용하겠습니다.
        """
        if self.adj_mat_mask != 'None':
            masked_loss_ce = nn.BCEWithLogitsLoss(weight=masked_mat)
        else:  # 마스크 없는 경우
            masked_loss_ce = self.loss_ce
        target_transposed = target.transpose(1, 2)  # 역방향 라벨
        loss1 = masked_loss_ce(scores1, target.float())
        loss2 = masked_loss_ce(scores2, target_transposed.float())
        return loss1, loss2

    def forward(self, pred_dict, tgt_dict):
        scores1, scores2 = pred_dict['scores1'], pred_dict['scores2']
        pred_htm, tgt_htm = pred_dict['pred_heatmaps'], tgt_dict['heatmaps']
        adj_mat_label, masked_mat, _ = self.matcher(pred_dict, tgt_dict)

        losses = {}
        losses['node'] = self.loss_node(pred_htm, tgt_htm)
        losses['graph1'], losses['graph2'] = self.loss_graph(
            scores1, scores2, adj_mat_label, masked_mat)

        losses['total'] = (losses['node'] * self.weight_dict['node'] +
                           losses['graph1'] * self.weight_dict['graph'] +
                           losses['graph2'] * self.weight_dict['graph'])

        return losses
