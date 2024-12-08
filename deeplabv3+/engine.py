import os
import time
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tensorboardX import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from model import create_deeplabv3

def denorm(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def visualize_predictions(images, true_masks, pred_masks, num_classes, save_path):
    """
    Visualize the original images, true masks, and predicted masks.

    Parameters:
    - images: Tensor of shape (batch_size, 3, height, width)
    - true_masks: Tensor of shape (batch_size, height, width)
    - pred_masks: Tensor of shape (batch_size, height, width)
    - num_classes: Number of classes
    - save_path: Path to save the visualized image
    """
    batch_size = images.shape[0]
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))

    # Define color map for visualization
    colors = plt.get_cmap('tab20', num_classes)

    for i in range(batch_size):
        image = denorm(images[i]).cpu().permute(1, 2, 0).numpy()
        true_mask = true_masks[i].cpu().numpy()
        pred_mask = pred_masks[i].cpu().numpy()

        axes[i, 0].imshow(image)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        true_colored_mask = colors(true_mask)
        axes[i, 1].imshow(image)
        axes[i, 1].imshow(true_colored_mask, alpha=0.5)
        axes[i, 1].set_title('True Mask')
        axes[i, 1].axis('off')

        pred_colored_mask = colors(pred_mask)
        axes[i, 2].imshow(image)
        axes[i, 2].imshow(pred_colored_mask, alpha=0.5)
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


class_names = [
    # 'void',
    'soil',
    'broad_bean',
    'corn_spurry',
    'red-root_amaranth',
    'common_buckwheat',
    'pea',
    'red_fingergrass',
    'common_wild_oat',
    'cornflower',
    'corn_cockle',
    'corn',
    'milk_thistle',
    'rye_brome',
    'soybean',
    'sunflower',
    'narrow-leaved_plantain',
    'small-flower_geranium',
    'sugar_beet'
]

# Create a mapping from class index to class name
class_indices = {i: name for i, name in enumerate(class_names)}

def compute_confusion_matrix(preds, labels, num_classes):
    preds = preds.flatten()
    labels = labels.flatten()
    mask = (labels >= 0) & (labels < num_classes)
    confusion = confusion_matrix(labels[mask], preds[mask], labels=np.arange(num_classes))
    return confusion

def compute_miou(confusion):
    intersection = np.diag(confusion)
    ground_truth_set = confusion.sum(axis=1)
    predicted_set = confusion.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    iou = intersection / union
    return np.nanmean(iou), iou

def evaluate_model(model, dataloader, device, num_classes):
    model.eval()
    confusion_matrix_all = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    with torch.no_grad():
        for img, mask, _ in dataloader:
            img = img.to(device)
            mask = mask.to(device)
            outputs = model(img)['out']
            preds = outputs.argmax(dim=1).cpu().numpy()
            mask = mask.cpu().numpy()

            for pred, true_mask in zip(preds, mask):
                # Print class indices in the original mask and their corresponding names
                unique_classes = np.unique(true_mask)
                confusion = compute_confusion_matrix(pred, true_mask, num_classes)
                confusion_matrix_all += confusion

    mean_iou, class_iou = compute_miou(confusion_matrix_all)
    return mean_iou, class_iou

class Trainer:
    def __init__(self, args, model, train_loader, val_loader, device):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epoch = 0
        self.iter = 0
        self.best_eval_mIOU = 0
        self.best_eval_mIOU_epoch = 0

        os.makedirs(args.output_dir, exist_ok=True)
        self.writer = SummaryWriter(args.output_dir)

        self.criterion = nn.CrossEntropyLoss(ignore_index = 255)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        os.makedirs(f'{self.args.output_dir}/{self.args.project_name}/ckpt/', exist_ok=True)

    def train_one_epoch(self):
        self.model.train()
        batch_loss = 0.0
        len_train = len(self.train_loader)

        pbar = tqdm(self.train_loader)
        for img, mask, target in pbar:
            self.iter += 1
            self.optimizer.zero_grad()

            img = img.to(self.device)
            mask = mask.to(self.device)
            outputs = self.model(img)['out']
            loss = self.criterion(outputs, mask)
            loss.backward()
            self.optimizer.step()

            batch_loss += loss.item()
            
            if self.iter % 10 == 0:
                pbar.set_description(f"Iter {self.iter}, Train loss: {loss.item():.3f}")

        avg_loss = batch_loss / len_train
        self.writer.add_scalar('data/average_loss', avg_loss, self.epoch)
        self.epoch += 1

        return avg_loss

    def visualization(self):
        self.model.eval()

        img, mask, _ = next(iter(self.train_loader))
        with torch.no_grad():
            img = img.to(self.device)
            mask = mask.to(self.device)

            outputs = self.model(img)['out']
            pred_mask = outputs.argmax(dim=1)

        os.makedirs(f'{self.args.output_dir}/{self.args.project_name}/figs/', exist_ok=True)
        visualize_predictions(img, mask, pred_mask, self.args.num_classes, f'{self.args.output_dir}/{self.args.project_name}/figs/epoch_{self.epoch}.jpg')

    def evaluation(self):
        
        mean_iou, class_iou = evaluate_model(self.model, self.val_loader, self.device, 18)
        print(f"Test Mean IOU: {mean_iou}")
        for i, iou in enumerate(class_iou):
            if i != 255:  # Skip void class
                print(f"Class {i} ({class_indices[i]}) IOU: {iou}")

        # class_iou_dict ={class_indices[i]: class_iou[i] for i in range(len(class_indices))}
            
        # self.writer.add_scalar('data/mIOU', mean_iou, self.epoch)
        # self.writer.add_scalars('data/class_ious', class_iou_dict, self.epoch)
        return mean_iou, class_iou

    def get_checkpoint(self):
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'iteration': self.iter,
            'best_eval_mIOU': self.best_eval_mIOU,
            'best_eval_mIOU_epoch': self.best_eval_mIOU_epoch,
        }
