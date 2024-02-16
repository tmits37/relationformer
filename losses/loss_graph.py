from .losses import SetCriterion


class GraphCriterion(SetCriterion):
    def __init__(self, config, matcher, net, distributed):
        super(SetCriterion, self).__init__(self, config, matcher, net, distributed)

    def forward(self, ):



    