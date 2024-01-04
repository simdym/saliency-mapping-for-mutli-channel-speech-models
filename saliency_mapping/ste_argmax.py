import torch
import torch.nn as nn
import torch.nn.functional as F

from examples.textless_nlp.dgslm.dgslm_utils import ApplyKmeans

class STEGumbelSoftmaxQuantizer(ApplyKmeans):
    """Quantizer with straight-through estimator gumbel softmax."""
    def __init__(self, km_path: str, num_classes: int, temperature: float=1.0):
        super().__init__(km_path)
        self.temperature = temperature
        self.num_classes = num_classes
    
    def __call__(self, x):
        """Quantize input with gumbel softmax.
        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_features).
        
        Returns:
            torch.Tensor: Quantized input as onehot-vector of shape (batch_size, seq_len, num_classes).
        """
        distances  = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
        
        zero_padding = 10000*torch.ones(distances.size(0), self.num_classes - distances.size(1)).cuda()
        distances = torch.cat([zero_padding, distances], dim=-1)

        print("distances", torch.argmax(-distances, dim=-1), torch.max(-distances, dim=-1))

        return F.gumbel_softmax(-distances, tau=self.temperature, dim=-1, hard=False)
