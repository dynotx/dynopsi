from typing import Any

import torch.nn as nn


class ProteinModule(nn.Module):
    def __init__(self, embedder: nn.Module, main_trunk: nn.Module, output_head: nn.Module):
        super().__init__()
        self.embedder = embedder
        self.main_trunk = main_trunk
        self.output_head = output_head

    def forward(self, x: dict[str, Any]) -> dict[str, Any]:
        emb = self.embedder(x)
        main_trunk_out = self.main_trunk(emb)
        output_head_out = self.output_head(main_trunk_out)
        return output_head_out
