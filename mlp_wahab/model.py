"""
MLP predictor for early exit decisions.

This is an exact copy of the MLP class from SpecEE_cloud/model_llama_ee.py (lines 949-963).
It must remain identical so that torch.load() in SpecEE can unpickle our trained models.

Architecture: 2-layer MLP with ReLU hidden activation and Sigmoid output.
    Input:  12 features (4 draft logits + 4 softmax probs + 4 prob gaps)
    Hidden: 512 units (matches original SpecEE predictors)
    Output: 1 scalar in [0, 1] — probability that the model can safely exit at this layer
"""

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
