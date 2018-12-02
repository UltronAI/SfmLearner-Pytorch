import argparse, time, csv, os, datetime

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import pose_transforms

from scipy.misc import imresize
import numpy as np
from path import Path
import argparse

from utils import tensor2array, save_checkpoint
from models import QuantPoseExpNet
from inverse_warp import pose_vec2mat
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter
from collections import OrderedDict
from datasets.pose_estimation import pose_framework_KITTI

parser = argparse.ArgumentParser(description="Script for supervised-training of PoseNet with corresponding groundtruth from KITTI Odometry.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--checkpoint', default=None, type=str, help="path to pretrained pose net model")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)

parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--train-sequences", default=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], 
                    type=str, nargs='*', help="sequences to train")
parser.add_argument("--test-sequences", default=['00'], type=str, nargs='*', help="sequences to test")
parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument('--dataset-format', default='sequential', metavar='STR',
                    help='dataset format, stacked: stacked frames (from original TensorFlow code) \
                    sequential: sequential folders (easier to convert to with a non KITTI/Cityscape dataset')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                    help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N', help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')

parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
# parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='csv where to save per-epoch train and valid stats')
# parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='csv where to save per-gradient descent train stats')

parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs and warped imgs at validation step')
parser.add_argument('-g', '--gpu-id', type=int, metavar='N', default=-1)
parser.add_argument('--use-scale-factor', action='store_true', help="use global scale factor")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    global best_error, n_iter, device
    args = parser.parse_args()

    save_path = save_path_formatter(args, parser)
    args.save_path = 'checkpoints'/save_path
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)

    training_writer = SummaryWriter(args.save_path)

    # Data loading code
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])
    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    print("=> fetching sequences in '{}'".format(args.dataset_dir))
    dataset_dir = Path(args.dataset_dir)
    print("=> preparing train set") 
    train_set = pose_framework_KITTI(
        dataset_dir, args.train_sequences, 
        sequence_length=args.sequence_length,
        transform=train_transform,
        seed=args.seed
    )
    print("=> preparing val set")
    val_set = pose_framework_KITTI(
        dataset_dir, args.test_sequences, 
        sequence_length=args.sequence_length,
        transform=valid_transform,
        seed=args.seed
    )
    print('{} samples found in {} train sequences'.format(len(train_set), train_set.sequence_num))
    print('{} samples found in {} valid sequences'.format(len(val_set), val_set.sequence_num))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")
    pose_net = QuantPoseExpNet(nb_ref_imgs=args.sequence_length-1, output_exp=False)
    if args.checkpoint is not None:
        print("=> using pre-trained weights for PoseNet")
        weights = torch.load(args.checkpoint)
        pose_net.load_state_dict(weights['state_dict'], strict=False)
    else:
        pose_net.init_weights()

    cudnn.benchmark = True
    if args.gpu_id == 0 or args.gpu_id == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    else:
        pose_net = torch.nn.DataParallel(pose_net)

    print('=> setting adam solver')

    optim_params = [
        {'params': pose_net.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
    logger.epoch_bar.start()

    if args.checkpoint:
        logger.reset_valid_bar()
        errors, error_names = validate(args, val_loader, pose_net)
        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, 0)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()
        train_loss = train(args, train_loader, pose_net, optimizer, args.epoch_size, logger, training_writer)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        logger.reset_valid_bar()
        errors, error_names = validate(args, train_loader, pose_net, optimizer, args.epoch_size)
        error_string = ', '.join('{} : {:.4f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)

        ate_error = errors[0]
        if best_error < 0:
            best_error = ate_error

        # remember lowest error and save checkpoint
        is_best = ate_error < best_error
        best_error = min(best_error, ate_error)

        if args.gpu_id < 0:
            save_checkpoint(
                args.save_path, {
                    'epoch': epoch + 1,
                    'state_dict': pose_exp_net.module.state_dict()
                },
                is_best)
        else:
            save_checkpoint(
                args.save_path, {
                    'epoch': epoch + 1,
                    'state_dict': pose_exp_net.state_dict()
                },
                is_best)

    logger.epoch_bar.finish()

def save_path_formatter(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['dataset_dir']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['epoch_size'] = 'epoch_size'
    keys_with_prefix['sequence_length'] = 'seq'
    keys_with_prefix['rotation_mode'] = 'rot_'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['use_scale_factor'] = 'use_scale_factor_'
    keys_with_prefix['lr'] = 'lr'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path(','.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    return save_path/timestamp


def train(args, train_loader, pose_net, optimizer, epoch_size, logger, train_writer):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)

    pose_net.train()
    end = time.time()
    logger.train_bar.update(0)

    for i, (imgs, groundtruth) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        h, w, _ = imgs[0].shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            imgs = [imresize(img, (args.img_height, args.img_width)).astype(np.float32) for img in imgs]
        imgs = [np.transpose(img, (2,0,1)) for img in imgs]

        ref_imgs = []
        for i, img in enumerate(imgs):
            img = torch.from_numpy(img).unsqueeze(0).to(device)
            if i == len(imgs)//2:
                tgt_img = img
            else:
                ref_imgs.append(img)

        # compute output
        _, poses = pose_net(tgt_img, ref_imgs)
        groundtruth = groundtruth.to(device)
        loss = compute_loss(groundtruth, poses, use_scale_factor)

        if i > 0 and n_iter % args.print_freq == 0:
            train_writer.add_scalar('Smooth L1 Loss', loss.item(), n_iter)

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]


def compute_loss(gt, pred, use_scale_factor=False):
    global device
    poses = pred.cpu()[0]
    poses = torch.cat([poses[:len(imgs)//2], torch.zeros(1,6).float(), poses[len(imgs)//2:]])
    inv_transform_matrices = pose_vec2mat(poses, rotation_mode=args.rotation_mode).numpy().astype(np.float64)

    rot_matrices = np.linalg.inv(inv_transform_matrices[:,:,:3])
    tr_vectors = -rot_matrices @ inv_transform_matrices[:,:,-1:]

    transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)

    first_inv_transform = inv_transform_matrices[0]
    final_poses = first_inv_transform[:,:3] @ transform_matrices
    final_poses[:,:,-1:] += first_inv_transform[:,-1:]
    final_poses = torch.from_numpy(final_poses).to(device)

    if use_scale_factor:
        np_poses = final_poses.cpu().numpy()
        np_gt = gt.cpu().numpy()
        scale_factor = np.sum(np_gt[:,:,-1] * np_poses[:,:,-1]) / np.sum(np_poses[:,:,-1] ** 2)
        loss = F.smooth_l1_loss(final_poses * scale_factor, gt)
    else:
        loss = F.smooth_l1_loss(final_poses, gt)

    return loss


@torch.no_grad()
def validate(args, val_loader, pose_net, logger):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=3, precision=4)

    pose_net.eval()
    end = time.time()
    logger.valid_bar.update(0)
    for i, (imgs, path, groundtruth) in enmuerate(val_loader):
        h, w, _ = imgs[0].shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            imgs = [imresize(img, (args.img_height, args.img_width)).astype(np.float32) for img in imgs]
        imgs = [np.transpose(img, (2,0,1)) for img in imgs]

        ref_imgs = []
        for i, img in enumerate(imgs):
            img = torch.from_numpy(img).unsqueeze(0).to(device)
            if i == len(imgs)//2:
                tgt_img = img
            else:
                ref_imgs.append(img)

        # compute output
        _, poses = pose_net(tgt_img, ref_imgs)

        poses = poses.cpu()[0]
        poses = torch.cat([poses[:len(imgs)//2], torch.zeros(1,6).float(), poses[len(imgs)//2:]])
        inv_transform_matrices = pose_vec2mat(poses, rotation_mode=args.rotation_mode).numpy().astype(np.float64)

        rot_matrices = np.linalg.inv(inv_transform_matrices[:,:,:3])
        tr_vectors = -rot_matrices @ inv_transform_matrices[:,:,-1:]

        transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)

        first_inv_transform = inv_transform_matrices[0]
        final_poses = first_inv_transform[:,:3] @ transform_matrices
        final_poses[:,:,-1:] += first_inv_transform[:,-1:]

        ATE, RE = compute_pose_error(groundtruth, final_poses)
        losses.update([ATE, RE])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))

    logger.valid_bar.update(len(val_loader))
    return losses.avg, ['ATE', 'RE']


def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    ATE = np.linalg.norm((gt[:,:,-1] - pred[:,:,-1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:,:3] @ np.linalg.inv(pred_pose[:,:3])
        s = np.linalg.norm([R[0,1]-R[1,0],
                            R[1,2]-R[2,1],
                            R[0,2]-R[2,0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s,c)

    return ATE/snippet_length, RE/snippet_length


def save_checkpoint(save_path, exp_pose_state, is_best, filename='checkpoint.pth.tar'):
    file_prefixes = ['exp_pose']
    states = [exp_pose_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}'.format(prefix,filename))

    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(save_path/'{}_{}'.format(prefix,filename), save_path/'{}_model_best.pth.tar'.format(prefix))

if __name__ == '__main__':
    main()
