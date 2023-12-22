import torch
import torch.nn.functional as F
from torch import nn
import itertools
import pdb
import box_ops_2D
import numpy as np


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.sum()/num_boxes

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class SetCriterion(nn.Module):
    """ This class computes the loss for Graphformer.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, config, matcher, net, distributed):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.net = net
        self.distributed = distributed
        self.losses = config.TRAIN.LOSSES
        self.weight_dict = {
            'node':config.TRAIN.W_NODE,
            'graph':config.TRAIN.W_GRAPH,
            }
        
    def loss_node(self, outputs, indices): # TODO 이걸 loss_node로 가야겠다
        """ 목적: 분류에 대한 손실을 계산합니다.
        작동 방식: outputs['pred_logits']를 사용하여 분류 손실을 계산합니다.
        이는 예측된 클래스 확률과 실제 클래스 레이블 사이의 크로스 엔트로피 손실입니다.
        특이사항: 가중치(weight)를 사용하여 특정 클래스에 대한 손실의 중요도를 조절합니다.
        """
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        weight = torch.tensor([0.2, 0.8]).to(outputs.get_device())
        
        idx = self._get_src_permutation_idx(indices)

        # targets = torch.zeros(outputs.shape[:-1], dtype=outputs.dtype).to(outputs.get_device())
        # targets[idx] = 1.0
        
        # targets = targets.unsqueeze(-1)
        
        # num_nodes = targets.sum()
        # # loss = F.cross_entropy(outputs.permute(0,2,1), targets, weight=weight, reduction='mean')
        # loss = sigmoid_focal_loss(outputs, targets, num_nodes)

        targets = torch.zeros(outputs[...,0].shape, dtype=torch.long).to(outputs.get_device())
        targets[idx] = 1.0
        loss = F.cross_entropy(outputs.permute(0,2,1), targets, weight=weight, reduction='mean')
        # cls_acc = 100 - accuracy(outputs, targets_one_hot)[0]



        # loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]).cuda())




        return loss
    
    def loss_graph(self, outputs, targets, indices):
        # TODO 로스 짜기 -> 검증하기
        # 어차피 gt가 1인 경우면 맞냐 틀리냐 로스로 들어감
        # gt에 대한 indices를 모아놓으면 시간복잡도를 낮출 수 있어보임
        # 모든 gt indices에 대해 맞냐 틀리냐 Sum 시키는 코드 만들기

        # sum_pred_values = sum(outputs[i, j] for i, j in indices)
        loss = 0
        for i, j in indices:
            loss += F.cross_entropy(outputs[i].unsqueeze(0), torch.tensor([j]))
        return loss / len(outputs)
    
    def loss_cardinality(self, outputs, indices):
        """ 목적: 예측된 객체 수와 실제 객체 수 사이의 차이를 계산하는 카디널리티 손실을 계산합니다.
        특이사항: 이는 주로 로깅 목적으로 사용되며, 그래디언트를 전파하지 않습니다.
        """
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        idx = self._get_src_permutation_idx(indices)
        targets = torch.zeros(outputs[...,0].shape, dtype=torch.long).to(outputs.get_device())
        targets[idx] = 1.0
        
        tgt_lengths = torch.as_tensor([t.sum() for t in targets], device=outputs.device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (outputs.argmax(-1) == outputs.shape[-1] - 1).sum(1)
        # card_pred = (outputs.sigmoid()>0.5).squeeze(-1).sum(1)

        loss = F.l1_loss(card_pred.float(), tgt_lengths.float(), reduction='sum')/(outputs.shape[0]*outputs.shape[1])

        return loss

    def loss_nodes_pos(self, outputs, targets, indices):
        """ 목적: 노드 위치에 대한 손실을 계산합니다.
        작동 방식: L1 손실을 사용하여 예측된 노드 위치와 실제 노드 위치 간의 차이를 계산합니다.
        """
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        num_nodes = sum(len(t) for t in targets)
        
        idx = self._get_src_permutation_idx(indices)
        pred_nodes = outputs[idx]
        target_nodes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss = F.l1_loss(pred_nodes, target_nodes, reduction='none') # TODO: check detr for loss function

        loss = loss.sum() / num_nodes

        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, out, target): # TODO out target I/O 따지기
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(out, target)
        
        losses = {}
        # losses['nodes'] = self.loss_nodes_pos(out['pred_nodes'][...,:2], target['nodes'], indices)
        losses['node'] = self.loss_node(out['pred_logits'], indices)
        losses['graph'] = self.loss_graph(out, target, indices)
        
        losses['total'] = sum([losses[key]*self.weight_dict[key] for key in self.losses])

        return losses