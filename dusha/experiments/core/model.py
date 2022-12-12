from typing import Union

import torch
from torch import nn
from torchvision.models.mobilenetv2 import ConvBNReLU, InvertedResidual, _make_divisible

AUDIO_PROBAS = ('audio_neg', 'audio_sad', 'audio_neu', 'audio_pos')
AUDIO_COLS = tuple(["audio_pred"] + list(AUDIO_PROBAS))

EMO2LABEL = {'angry': 0,
             'sad': 1,
             'neutral': 2,
             'positive': 3}


class SoftMaxModel(nn.Module):
    def __init__(self, logits_model: nn.Module):
        super().__init__()
        self.logits_model = logits_model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.logits_model(x)
        x = self.softmax(x)

        return x


# slightly modified fast.ai implementation
# https://medium.com/mlearning-ai/self-attention-in-convolutional-neural-networks-172d947afc00
class ConvSelfAttention(nn.Module):
    """Self attention layer for `n_channels`."""

    def __init__(self, n_channels):
        super().__init__()
        self.query, self.key, self.value = [
            self._conv(n_channels, c)
            for c in (n_channels // 8, n_channels // 8, n_channels)
        ]
        self.gamma = nn.Parameter(torch.tensor([0.0]))

    def _conv(self, n_in, n_out):
        return nn.Conv1d(n_in, n_out, kernel_size=1, bias=False)

    def forward(self, x):
        # Notation from the paper.
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = nn.functional.softmax(torch.bmm(f.transpose(1, 2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


# see deep_pipe
# https://github.com/neuro-ml/deep_pipe/blob/master/dpipe/layers/shape.py#L48
class Reshape(nn.Module):
    """
    Reshape the incoming tensor to the given ``shape``.

    Parameters
    ----------
    shape: Union[int, str]
        the resulting shape. String values denote indices in the input tensor's shape.

    Examples
    --------
    >>> layer = Reshape('0', '1', 500, 500)
    >>> layer(x)
    >>> # same as
    >>> x.reshape(x.shape[0], x.shape[1], 500, 500)
    """

    def __init__(self, *shape: Union[int, str]):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor):
        shape = [x.shape[int(i)] if isinstance(i, str) else i for i in self.shape]
        return x.reshape(*shape)


# see torchvision.models.mobilenetv2.MobileNetV2
class ConvSelfAttentionMobileNet(nn.Module):
    def __init__(self, _config, n_classes, last_channel=128, in_channels=1):

        super().__init__()
        self._config = _config
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.last_channel = last_channel

        block = InvertedResidual
        norm_layer = nn.BatchNorm2d
        width_mult = 1.0
        round_nearest = 8

        input_channel = 4

        features = [
            ConvBNReLU(self.in_channels, input_channel, stride=1, norm_layer=norm_layer)
        ]
        for t, c, n, s in _config:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        norm_layer=norm_layer,
                    )
                )
                input_channel = output_channel
        # building last several layers
        features.append(
            ConvBNReLU(
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        self.pooling = nn.Sequential(
            ConvSelfAttention(self.last_channel),
            nn.AdaptiveAvgPool2d((1, 1)),
            Reshape("0", self.last_channel),
        )

        self.classifier = nn.Linear(self.last_channel, self.n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        x = self.classifier(x)

        return x
