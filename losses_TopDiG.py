import torch
import torch.nn.functional as F
from torch import nn
import itertools
import pdb
import numpy as np

class SetCriterion(nn.Module):
    """ This class computes the loss for Graphformer.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, config, net, distributed):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.net = net
        self.distributed = distributed
        self.losses = config.TRAIN.LOSSES
        self.weight_dict = { # 이 가중치도 조정가능
            'node':config.TRAIN.W_NODE,
            'graph':config.TRAIN.W_GRAPH,
            }
        self.loss_weighted_ce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100]).cuda()) # 100에서 조정가능
        self.loss_ce = nn.BCEWithLogitsLoss()
        
    def loss_node(self, output, target): # TODO
        """ pretrain처럼 PolyWorld와 같은 weighted BCE 사용하겠습니다
        """
        loss = self.loss_weighted_ce(output, target)
        return loss
    
    def loss_graph(self, output, target): # TODO 
        """ sigmoid로 scores 한칸 한칸 값의 범위를 (0~1) 스케일로 바꿔주겠다
        TopDiG처럼 BCE 사용하겠습니다
        """
        loss = self.loss_ce(output, target)
        return loss

    def forward(self, out, target): # TODO out target I/O 따지기
        losses = {}
        # 밑에 두개의 out과 target이 서로 다른 것 같다
        losses['node'] = self.loss_node(out, target)
        losses['graph'] = self.loss_graph(out, target)
        
        losses['total'] = sum([losses[key]*self.weight_dict[key] for key in self.losses])

        return losses