import torch
from torch import nn

from neuralnetwork.layers import ConvolutionalLayer, HourglassLayer, Pool, ResidualLayer
from neuralnetwork.losses.HeatmapLoss import HeatmapLoss
from models.utils.softargmax import softargmax2d


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.convolutional_layer = ConvolutionalLayer(x_dim, y_dim, 1,
                                                      rectified_linear_unit=False, batch_normalization=False)

    def forward(self, x):
        return self.convolutional_layer(x)


class EyeNet(nn.Module):
    def __init__(self, num_stack_of_hourglass, num_features, num_landmarks,
                 batch_norm=False, increase_feature=0, **kwargs):
        super(EyeNet, self).__init__()

        self.img_width = 160
        self.img_height = 96
        self.num_stacks = num_stack_of_hourglass
        self.num_features = num_features
        self.num_landmarks = num_landmarks

        self.heatmap_width = self.img_width / 2
        self.heatmap_height = self.img_height / 2

        self.num_stacks = num_stack_of_hourglass
        self.preprocess_layer = nn.Sequential(
            ConvolutionalLayer(1, 64, 7, 1,
                               batch_normalization=True, rectified_linear_unit=True),
            ResidualLayer(64, 128),
            Pool(2, 2),
            ResidualLayer(128, 128),
            ResidualLayer(128, num_features)
        )

        self.gaze_preprocess_layer = nn.Sequential(
            ConvolutionalLayer(num_features, 64, 7, 2,
                               batch_normalization=True, rectified_linear_unit=True),
            ResidualLayer(64, 128),
            Pool(2, 2),
            ResidualLayer(128, 128),
            ResidualLayer(128, num_features)
        )

        # a stack of 'stacks_of_hourglass' amount in hourglass modules
        self.hourglass_layer = nn.ModuleList([
            nn.Sequential(
                HourglassLayer(4, num_features, batch_norm, increase_feature),
            ) for i in range(num_stack_of_hourglass)])

        self.feature_layer = nn.ModuleList([
            nn.Sequential(
                ResidualLayer(num_features, num_features),
                ConvolutionalLayer(num_features, num_features, 1,
                                   batch_normalization=True, rectified_linear_unit=True)
            ) for i in range(num_stack_of_hourglass)])

        self.output_layer = nn.ModuleList([
            ConvolutionalLayer(num_features, num_landmarks, 1,
                               rectified_linear_unit=False,
                               batch_normalization=False) for i in range(num_stack_of_hourglass)])
        self.merge_feature_layer = nn.ModuleList([
            Merge(num_features, num_features) for i in range(num_stack_of_hourglass - 1)
        ])
        self.merge_prediction_layer = nn.ModuleList([
            Merge(num_landmarks, num_features) for i in range(num_stack_of_hourglass - 1)
        ])

        self.gaze_prediction_full_connected_1 = nn.Linear(
            in_features=int(num_features * self.img_width * self.img_height / 64 + num_landmarks * 2),
            out_features=256)
        self.gaze_prediction_full_connected_2 = nn.Linear(in_features=256, out_features=2)

        self.num_stacks = num_stack_of_hourglass
        self.heatmapLoss = HeatmapLoss()
        self.landmarks_loss = nn.MSELoss()
        self.gaze_loss = nn.MSELoss()

    def forward(self, images):
        # images of size 1,ih,iw
        input_tensor = images.unsqueeze(1)
        input_tensor = self.preprocess_layer(input_tensor)

        gaze_tensor = self.gaze_preprocess_layer(input_tensor)
        gaze_tensor = gaze_tensor.flatten(start_dim=1)

        hourglass_module_predict_stack = []
        for i in torch.arange(self.num_stacks):
            hourglass_module = self.hourglass_layer[i](input_tensor)
            feature_refinement = self.feature_layer[i](hourglass_module)
            prediction_layer = self.output_layer[i](feature_refinement)
            hourglass_module_predict_stack.append(prediction_layer)
            if i < self.num_stacks - 1:
                input_tensor = (input_tensor
                                + self.merge_prediction_layer[i](prediction_layer)
                                + self.merge_feature_layer[i](feature_refinement))

        heatmaps_out = torch.stack(hourglass_module_predict_stack, 1)

        # preds = N x nlandmarks * heatmap_w * heatmap_h
        landmarks_out = softargmax2d(prediction_layer)  # N x nlandmarks x 2

        # Gaze
        gaze_prediction = torch.cat((gaze_tensor, landmarks_out.flatten(start_dim=1)), dim=1)
        gaze_prediction = self.gaze_prediction_full_connected_1(gaze_prediction)
        gaze_prediction = nn.functional.relu(gaze_prediction)
        gaze_prediction = self.gaze_prediction_full_connected_2(gaze_prediction)

        return heatmaps_out, landmarks_out, gaze_prediction

    def calc_loss(self, hourglass_module_predict_stack, heatmaps, landmarks_pred, landmarks, gaze_pred, gaze):
        hourglass_stack_with_loss = []
        for i in range(self.num_stacks):
            hourglass_stack_with_loss.append(self.heatmapLossFunction(hourglass_module_predict_stack[:, i, :], heatmaps))

        heatmap_loss = torch.stack(hourglass_stack_with_loss, dim=1)
        landmarks_loss = self.landmarks_loss(landmarks_pred, landmarks)
        gaze_loss = self.gaze_loss(gaze_pred, gaze)

        return torch.sum(heatmap_loss), landmarks_loss, 1000 * gaze_loss
