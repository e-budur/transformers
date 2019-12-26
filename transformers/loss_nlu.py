


from torch.nn import BCEWithLogitsLoss

class BCEWithLogitsLossNLU(BCEWithLogitsLoss):

    def __init__(self,
                 weight=None,
                 size_average=None,
                 reduce=None,
                 reduction='mean',
                 pos_weight=None,
                 ignore_negative_targets=False):
        super(BCEWithLogitsLossNLU, self).__init__(weight=weight,
                                                size_average=size_average,
                                                reduce=reduce,
                                                reduction=reduction,
                                                pos_weight=pos_weight)
        self.ignore_negative_targets = ignore_negative_targets

    def forward(self, input, target):
        if self.ignore_negative_targets:
            non_negative_target_indexes = target>=0.0
            input = input[non_negative_target_indexes]
            target = target[non_negative_target_indexes]
        loss = super(BCEWithLogitsLossNLU, self).forward(input, target)
        return loss
