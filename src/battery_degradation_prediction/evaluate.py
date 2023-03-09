import torch
from torch import nn


def evaluate(
    model: nn.Module, eval_data: torch.Tensor, targets: torch.Tensor, criterion
) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(eval_data.size(0)):
            # data, targets = get_batch(eval_data, i)
            # seq_len = data.size(0)
            output, _ = model(eval_data)
            total_loss = criterion(output, targets).item()
    return total_loss
