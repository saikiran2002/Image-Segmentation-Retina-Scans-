import torch
import torch.distributed as dist
import sys
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm import tqdm
device = torch.device('cuda')
# Class to compute the confusion matrix
class ConfusionMatrix:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    # Update the confusion matrix
    def update(self, a, b):
        n = self.num_classes
        with torch.no_grad():
            mask = (a >= 0) & (a < n)
            a = a[mask].to(torch.int64)
            b = b[mask].to(torch.int64)
            indices = n * a + b
            counts = torch.bincount(indices, minlength=n ** 2)
            self.mat += counts.reshape(n, n)

    # Reset the confusion matrix
    def reset(self):
        self.mat.zero_()

    # Compute metrics based on the confusion matrix
    def compute(self):
        h = self.mat.float()
        diag_sum = torch.diag(h).sum()
        total_sum = h.sum()
        acc_global = (diag_sum / total_sum).item() if total_sum > 0 else 0.0
        se = (h[1, 1] / h[1].sum()).item() if h[1].sum() > 0 else 0.0
        sp = (h[0, 0] / h[0].sum()).item() if h[0].sum() > 0 else 0.0
        pr = (h[1, 1] / h[:, 1].sum()).item() if h[:, 1].sum() > 0 else 0.0
        F1 = 2 * (pr * se) / (pr + se) if (pr + se) > 0 else 0.0
        return acc_global, se, sp, F1, pr

# Function to evaluate the model
def evaluate(model, data_loader, num_classes):
    model.eval()
    confmat = ConfusionMatrix(num_classes, device="cuda")
    data_loader = tqdm(data_loader)
    mask = None
    predict = None
    dice_c = 0

    with torch.no_grad():
        for data in data_loader:
            for image, target, eye_masks in data:
                image, target = image.cuda(), target.cuda()
                output = model(image)
                if type(output) is list:
                    output = output[0]
                output = torch.sigmoid(output)
                truth = output.clone()
                output[output >= 0.5] = 1
                output[output < 0.5] = 0
                confmat.update(target.flatten(), output.long().flatten())
                dice_c += dice_coeff(output, target)
                mask = target.flatten() if mask is None else torch.cat((mask, target.flatten()))
                predict = truth.flatten() if predict is None else torch.cat((predict, truth.flatten()))

    mask = mask.cpu().numpy()
    predict = predict.cpu().numpy()
    AUC_ROC = roc_auc_score(mask, predict)
    iou = calculate_iou(model, data_loader)
    return confmat.compute()[0], confmat.compute()[1], confmat.compute()[2], confmat.compute()[3], confmat.compute()[
        4], AUC_ROC, dice_c / len(data_loader), iou

# Function to compute the Dice coefficient
def dice_coeff(x: torch.Tensor, target: torch.Tensor, epsilon=1e-6):
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1).float()
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter
        d += (2 * inter + epsilon) / (sets_sum + epsilon)
    return d / batch_size

# Function to calculate the Intersection over Union (IOU)
def calculate_iou(model, dataloader):
    model.eval()
    total_iou = 0.0
    num_batches = len(dataloader)
    with torch.no_grad():
        for data in dataloader:
            z = len(data)
            for images, masks, eye_masks in data:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                
                # Convert outputs to binary predictions
                binary_preds = (outputs > 0.5).float()
                
                # Calculate intersection and union
                intersection = torch.logical_and(binary_preds, masks).sum((1, 2))  # Sum over height and width
                union = torch.logical_or(binary_preds, masks).sum((1, 2))         # Sum over height and width
                
                # Calculate IOU for each sample in the batch
                iou_per_sample = torch.where(union == 0, torch.ones_like(union), intersection.float() / union.float())
                
                # Average IOU across the batch
                batch_iou = iou_per_sample.mean().item()
                total_iou += batch_iou
    
    # Calculate average IOU across all batches
    avg_iou = total_iou / (num_batches*z)
    return avg_iou

if __name__ == "__main__":
    print("This is a metric file")