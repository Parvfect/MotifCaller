
import torch
import torch.nn as nn


class GreedyCTCDecoder(nn.Module):

    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels=labels
        self.blank = blank

    def forward(self, emission:torch.Tensor):
        """Given a sequence emission over labels, get the best path"""

        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = " ".join([self.labels[i] for i in indices])
        return joined.replace("|", " ").strip().split()
    
    def forward_2(self, emission, prob_threshold):
        probs = torch.exp(emission)  # shape: (T, C)
        max_probs, indices = torch.max(probs, dim=-1)  # get max prob and corresponding index at each timestep

        # Apply probability threshold
        indices = torch.where(max_probs >= prob_threshold, indices, torch.tensor(self.blank, device=indices.device))

        # Collapse repeated tokens and remove blanks
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]

        # Convert to label string
        joined = " ".join([self.labels[i] for i in indices])
        return joined.replace("|", " ").strip().split()
    