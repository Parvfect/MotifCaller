
#from torchaudio.models.decoder import ctc_decoder
import torch
import torch.nn as nn
import numpy as np


class GreedyCTCDecoder(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        labels_int = np.arange(n_classes).tolist()
        self.labels = [f"{i}" for i in labels_int]
        self.blank = 0

    def forward(self, emission:torch.Tensor):
        """Given a sequence emission over labels, get the best path"""

        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = " ".join([self.labels[i] for i in indices])
        return joined.replace("|", " ").strip().split()

""" Ignoring beam decoder for now
def torch_ctc(n_classes, model_output, beam_width=5, metrics=False):

    labels = ["|"] + ["-"] + [str(i) for i in range(n_classes)]
    decoder = ctc_decoder(lexicon=None, tokens=labels, beam_size=beam_width)
    output = decoder(model_output)
    tokens = output[0][0].tokens.tolist()

    if not metrics:
        return [i for i in tokens if i > 0]
    
    return output
"""