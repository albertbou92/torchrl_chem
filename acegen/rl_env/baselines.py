import torch


class MovingAverageBaseline:
    """Class to keep track on the running mean and variance of tensors batches."""

    def __init__(self, epsilon=1e-3, shape=(), device=torch.device("cpu")):
        self.mean = torch.zeros(shape, dtype=torch.float64).to(device)
        self.std = torch.zeros(shape, dtype=torch.float64).to(device)
        self.count = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_std = torch.std(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_std, batch_count)

    def update_from_moments(self, batch_mean, batch_std, batch_count):
        delta = batch_mean - self.mean
        std_delta = batch_std - self.std
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        new_std = self.std + std_delta * batch_count / tot_count
        new_count = tot_count
        self.mean, self.std, self.count = new_mean, new_std, new_count


class LeaveOneOutBaseline:
    """Class to compute the leave-one-out baseline for a given tensor."""

    def __init__(self):
        self.mean = None

    def update(self, x):
        with torch.no_grad():
            loo = x.unsqueeze(1).expand(-1, x.size(0))
            loo_mask = 1 - torch.eye(loo.size(0), device=loo.device)
            self.mean = (loo * loo_mask).sum(0) / loo_mask.sum(0)
