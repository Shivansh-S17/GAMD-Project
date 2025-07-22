from colorama import Fore
import torch
import os
import numpy as np
# Accuracy check function


def pointnet_regularization(trans):
    I = torch.eye(trans.size(-1), trans.size(-1))
    I = I.cuda()
    loss = torch.mean(torch.norm(
        torch.bmm(trans, trans.transpose(-1, -2)) - I, dim=(1, 2)))
    return loss

# Checkpoint saving function

def save_checkpoint(state, filename,run_name,extension):
    x = os.path.join(filename,run_name+extension)
    if os.path.exists(x):
        torch.save(state, x)
        print(Fore.GREEN+"Saving Checkpoint")
        return
    else:
        with open(x, 'w') as fp:
            fp.close()
        # If model is running for first time then Save
        torch.save(state, x)
        return

def meanIOU(true, predicted):
    # Calculates mIOU for a mini-batch
    predicted_arr = predicted.clone().detach().cpu().numpy().argmax(1)
    groundt = true.clone().detach().cpu().numpy()
    intersection = np.logical_and(groundt, predicted_arr).sum()
    union = np.logical_or(groundt, predicted_arr).sum()

    iou_score = intersection / union

    return iou_score
