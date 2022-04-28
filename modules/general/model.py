import torch
from .cpc import CPC


class Model(torch.nn.Module):
    def __init__(
        self, args, genc_hidden, gar_hidden,
    ):
        super(Model, self).__init__()

        self.args = args
        self.genc_input = args.genc_input
        self.genc_hidden = genc_hidden
        self.gar_hidden = gar_hidden

        self.model = CPC(
            args,
            self.genc_input,
            genc_hidden,
            gar_hidden,
        )

    def forward(self, x):
        """Forward through the network"""

        loss, accuracy, _, z = self.model(x)
        return loss
