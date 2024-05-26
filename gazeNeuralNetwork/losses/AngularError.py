import torch


class AngularError(torch.nn.Module):
    def __init__(self):
        super(AngularError, self).__init__()

    def forward(self, gaze_prediction, gaze):
        loss = ((gaze_prediction - gaze) ** 2)
        loss = torch.mean(loss, dim=(1, 2, 3))
        return loss
