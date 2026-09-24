import torch
import torch.distributed as dist

@torch.no_grad()
def class_pixel_counts(dataloader, tasks, num_classes, device, ignore_index=None):
    """Count pixels per class over a dataloader."""
    if isinstance(tasks, str):
        tasks = [tasks]
    if isinstance(num_classes, int):
        num_classes = [num_classes] * len(tasks)

    seg_idx = [i for i, t in enumerate(tasks) if t == "segmentation"]
    counts = {
        i: torch.zeros(num_classes[i], dtype=torch.long, device=device) for i in seg_idx
    }

    for batch in dataloader:
        for i in seg_idx:
            label = batch[i + 1]
            if label is None or label.numel() == 0:
                continue
            label = label.to(device, non_blocking=True)
            if label.dim() == 4 and label.size(1) == num_classes[i]:  # one-hot masks
                label = label.argmax(dim=1)
            label = label.long().reshape(-1)
            if ignore_index is not None:
                label = label[label != ignore_index]
            bc = torch.bincount(label, minlength=num_classes[i])
            if bc.numel() > num_classes[i]:
                raise ValueError(
                    f"Task {i}: found label {bc.numel() - 1}, "
                    f"but num_classes is {num_classes[i]}"
                )
            counts[i] += bc
    
    if dist.is_available() and dist.is_initialized():
        for i in seg_idx:  # same order on every rank
            dist.all_reduce(counts[i], op=dist.ReduceOp.SUM)
    
    for i in seg_idx:
        if counts[i].sum() == 0:
            print(f"Warning: no labels found for segmentation task {i}")

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