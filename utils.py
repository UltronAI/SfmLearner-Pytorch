from __future__ import division
import shutil
import numpy as np
import torch
from path import Path
import datetime
from collections import OrderedDict
from math import floor, log2

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def save_path_formatter(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['epoch_size'] = 'epoch_size'
    keys_with_prefix['sequence_length'] = 'seq'
    keys_with_prefix['rotation_mode'] = 'rot_'
    keys_with_prefix['padding_mode'] = 'padding_'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'
    keys_with_prefix['photo_loss_weight'] = 'p'
    keys_with_prefix['mask_loss_weight'] = 'm'
    keys_with_prefix['smooth_loss_weight'] = 's'
    keys_with_prefix['use_disp'] = 'use_disp_'
    keys_with_prefix['smooth_loss_factor'] = 'scale_factor_'
    keys_with_prefix['use_quant_model'] = 'use_quant_model_'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path(','.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    return save_path/timestamp


def tensor2array(tensor, max_value=255, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            import cv2
            if int(cv2.__version__[0]) >= 3:
                color_cvt = cv2.COLOR_BGR2RGB
            else:  # 2.4
                color_cvt = cv2.cv.CV_BGR2RGB
            if colormap == 'rainbow':
                colormap = cv2.COLORMAP_RAINBOW
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = (255*tensor.squeeze().numpy()/max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32)/255
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy()/max_value).clip(0,1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy()*0.5
    return array


def save_checkpoint(save_path, dispnet_state, exp_pose_state, is_best, filename='checkpoint.pth.tar'):
    file_prefixes = ['dispnet', 'exp_pose']
    states = [dispnet_state, exp_pose_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}'.format(prefix,filename))

    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(save_path/'{}_{}'.format(prefix,filename), save_path/'{}_model_best.pth.tar'.format(prefix))

def getScale(_arr):
    arr = np.copy(_arr)
    arr_max, arr_min = np.max(arr), np.min(arr)
    scale = max(
                floor(log2(abs(arr_max if arr_max > 0 else arr_max + 1))) + 1,
                floor(log2(abs(arr_min if arr_min > 0 else arr_min + 1))) + 1
                )
    scale = log2(2 ** 7 / 2 ** scale)
    return scale

def quantize(_arr):
    """
    input: numpy array
    output: 8-bit quantized numpy array
    """
    assert type(_arr) is np.ndarray, 'you need convert torch to np.ndarray'

    arr = np.copy(_arr)
    scale = getScale(arr)
    quant_num = np.arange(-128 * 2 ** (-scale), 128 * 2 ** (-scale), 2 ** (-scale))
    for ele in np.nditer(arr, op_flags=['readwrite']):
        ele[...] =  min(quant_num, key=lambda x: abs(x - ele))

    return arr, scale

def quantizeWeight(state_dict, fix_info):
    new_state_dict = {}
    for key, value in state_dict.items():
        cur_value = value.cpu().numpy().copy()
        quant_value, scale = quantize(cur_value)
        fix_key = '.'.join(key.split('.')[:-1])
        if fix_key in fix_info.keys():
            fix_info[fix_key][key.split('.')[-1]] = scale
        else:
            fix_info[fix_key] = {}
            fix_info[fix_key][key.split('.')[-1]] = scale
        new_state_dict[key] = torch.from_numpy(quant_value).to(device)

    return new_state_dict

