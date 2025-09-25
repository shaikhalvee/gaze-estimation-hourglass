import torch


class HeatmapLoss(torch.nn.Module):
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, predicate, ground_truth):
        loss = (predicate - ground_truth) ** 2
        loss = torch.mean(loss, dim=(1, 2, 3))
        return loss
