import torch
import torch.distributed as dist

@torch.no_grad()
def class_pixel_counts(loader, num_classes, device, ignore_index=None):
    """Count pixels per class over a (possibly distributed) dataloader."""
    counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    for batch in loader:
        targets = batch[1]  # adapt if your batch is a dict / different order
        targets = targets.to(device, non_blocking=True)
        if targets.dim() == 4 and targets.size(1) == num_classes:  # one-hot masks
            targets = targets.argmax(dim=1)
        targets = targets.long().reshape(-1)
        if ignore_index is not None:
            targets = targets[targets != ignore_index]
        bc = torch.bincount(targets, minlength=num_classes)
        assert bc.numel() == num_classes, f"Found label >= num_classes ({bc.numel()})"
        counts += bc
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return counts


def class_weights(counts, scheme="inverse_sqrt", eps=1e-12):
    """Turn pixel counts into class weights, normalized to mean 1."""
    freq = counts.double() / counts.sum().clamp_min(1)
    freq = freq.clamp_min(eps)
    if scheme == "inverse":          # N / (C * n_c)
        w = 1.0 / freq
    elif scheme == "inverse_sqrt":   # softer version
        w = 1.0 / freq.sqrt()
    elif scheme == "enet":           # Paszke et al. (ENet): 1 / ln(c + p)
        w = 1.0 / torch.log(1.02 + freq)
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")
    w = w * (len(w) / w.sum())
    return w.float()