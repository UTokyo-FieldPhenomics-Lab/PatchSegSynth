import time

import torch

from engine import Trainer

import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import WE3DSDataset
from model import create_deeplabv3

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
# from options_base_synthetic import get_arguments
# from options_distribution_synthetic import get_arguments

import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description="Training script for DeepLabV3+ model")

    parser.add_argument('--project_name', type=str, default='patch_level_5x', help='Project name for outputs')
    parser.add_argument('--train_images_dir', type=str, default='../WE3DS_DATASET/Synthetic/scenes_10x_old/images', help='Directory for training images')
    parser.add_argument('--train_segmentations_dir', type=str, default='../WE3DS_DATASET/Synthetic/scenes_10x_old/annotations', help='Directory for training segmentations')

    parser.add_argument('--test_images_dir', type=str, default='../WE3DS_DATASET/Test/images', help='Directory for test images')
    parser.add_argument('--test_segmentations_dir', type=str, default='../WE3DS_DATASET/Test/annotations', help='Directory for test segmentations')

    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--num_classes', type=int, default=18, help='Number of classes for segmentation')
    parser.add_argument('--input_size', type=int, nargs=2, default=(640, 480), help='Input image size')


    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use for training')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save outputs')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default='', help='Path to resume checkpoint')
    parser.add_argument('--alpha', type=float, default=0.9, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta', type=float, default=0.999, help='Beta2 for Adam optimizer')
    parser.add_argument('--eval_freq', type=int, default=5, help='Frequency of evaluation')
    parser.add_argument('--vis_freq', type=int, default=1, help='Frequency of visualization')
    parser.add_argument('--save_freq', type=int, default=5, help='Frequency of saving checkpoints')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    transform_train = A.Compose(
        [   
            A.Resize(480, 640),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    transform_val = A.Compose(
        [   
            A.Resize(480, 640),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    num_classes = 18

    train_dataset = WE3DSDataset(args.train_images_dir, args.train_segmentations_dir, transform=transform_train, train=True)
    val_dataset = WE3DSDataset(args.test_images_dir, args.test_segmentations_dir, transform=transform_val, train=False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8)

    model = create_deeplabv3(num_classes).to(device)
    engine = Trainer(args, model, train_loader, val_loader, device)

    print(f'using device: {engine.device}')

    project_dir = f'{args.output_dir}/{args.project_name}'
    ckpt_dir = f'{project_dir}/ckpt'

    best_eval_mIOU = engine.best_eval_mIOU
    ## Start Training ##
    # Main loop
    for _ in range(engine.epoch, engine.epoch + args.epochs):
        # always run evaluation first (to check what model has been loaded !!!)
        if engine.epoch == 0 or args.resume != '':
            engine.visualization()
            # t1 = time.time()
            # mean_miou, class_miou = engine.evaluation()
            # t2 = time.time()
            # # print the evaluation results
            # print('=======================================Evaluation=======================================')
            # print(f"[Epoch {engine.epoch}], time: {(t2-t1):.2f}s, best mIOU: {engine.best_eval_mIOU:.4f}, at epoch: {engine.best_eval_mIOU_epoch}")
            # print(f"Class mIoU: {class_miou}")
            # print(f"Mean mIoU: {mean_miou:.4f}")
            # print('=======================================Evaluation=======================================')

        # train
        t1 = time.time()
        avg_loss = engine.train_one_epoch()
        t2 = time.time()
        lr = engine.optimizer.param_groups[0]['lr']
        print(f'[epoch {engine.epoch}][lr {lr:.7f}][{(t2-t1):.2f}s]')
        
        # evaluation
        if engine.epoch % args.eval_freq == 0:
            t1 = time.time()
            mean_miou, class_miou = engine.evaluation()
            t2 = time.time()
            # print the evaluation results
            print('=======================================Evaluation=======================================')
            print(f"[Epoch {engine.epoch}], time: {(t2-t1):.2f}s, best mIOU: {engine.best_eval_mIOU:.4f}, at epoch: {engine.best_eval_mIOU_epoch}")
            print(f"Class mIoU: {class_miou}")
            print(f"Mean mIoU: {mean_miou:.4f}")
            print('=======================================Evaluation=======================================')
            if mean_miou > engine.best_eval_mIOU:
                print(f"Updating best model, best mIOU: {engine.best_eval_mIOU:.4f}, at epoch: {engine.best_eval_mIOU_epoch}")
                engine.best_eval_mIOU = mean_miou
                engine.best_eval_mIOU_epoch = engine.epoch
                # engine.best_model = copy.deepcopy(engine.get_checkpoint())
                torch.save(engine.get_checkpoint(), f'{ckpt_dir}/epoch_{engine.epoch}_{engine.best_eval_mIOU:.4f}.pth')
                

        # visualization
        if engine.epoch % args.vis_freq == 0:
            engine.visualization()

        # save model checkpoints
        if engine.epoch % args.save_freq == 0:
            print(f'saving checkpoints @epoch {engine.epoch}')
            torch.save(engine.get_checkpoint(), f'{ckpt_dir}/epoch_{engine.epoch}.pth')

        # torch.save(engine.best_model, f'{ckpt_dir}/epoch_{engine.best_eval_mIOU_epoch}_eval_mIOU_{engine.best_eval_mIOU:.4f}.pth')
    
    print(f'Training process finished, best eval mIOU: {engine.best_eval_mIOU:.4f}')
    # torch.save(engine.best_model, f'{ckpt_dir}/best_eval_mIOU_{engine.best_eval_mIOU:.4f}.pth')
    engine.writer.close()
