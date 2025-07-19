from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.utils import ModelOutput


@dataclass
class BertModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    scores: torch.FloatTensor = None