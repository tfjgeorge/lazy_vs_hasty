import numpy as np

def margin_mean(logits, targets):
    logits_target = logits[np.arange(len(logits)), targets]
    logits_cp = logits.clone()

    logits_cp[np.arange(len(logits)), targets] = 0
    margins = logits_target - logits_cp.sum(dim=1) / 9

    return margins
