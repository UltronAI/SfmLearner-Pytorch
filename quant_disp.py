import torch

from imageio import imread, imsave
from scipy.misc import imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from models import QuantDispNetS
from utils import tensor2array, getScale, quantize, quantizeWeight

parser = argparse.ArgumentParser(description='Quantize Pretrained DispNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-disp", action='store_true', help="save disparity img")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")

parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--quantize-weights", action='store_true')
parser.add_argument("--original-input", action='store_true')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

fix_info = {}

def quantHookWithName(module_name):
    def quantHook(module, input, output):
        if 'input' not in fix_info[module_name].keys():
            fix_info[module_name]['input'] = getScale(input[0].cpu().numpy().copy())
        else:
            fix_info[module_name]['input'] = min(fix_info[module_name]['input'], getScale(input[0].cpu().numpy().copy()))
        if 'output' not in fix_info[module_name].keys():
            fix_info[module_name]['output'] = getScale(output[0].cpu().numpy().copy())
        else:
            fix_info[module_name]['output'] = min(fix_info[module_name]['output'], getScale(output[0].cpu().numpy().copy()))
    return quantHook

@torch.no_grad()
def main():
    args = parser.parse_args()
    if not(args.output_disp or args.output_depth):
        print('You must at least output one value !')
        # return

    disp_net = QuantDispNetS().to(device)
    weights = torch.load(args.pretrained)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    if args.quantize_weights:
        print("Quantizing the pretrained model ...")
        quant_weights = { 'state_dict': quantizeWeight(weights['state_dict'], fix_info) }
        torch.save(quant_weights, output_dir/'quant_dispnet_model.pth.tar')
    else:
        for key in weights['state_dict'].keys():
            fix_info['.'.join(key.split('.')[:-1])] = {}

    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = [dataset_dir/file for file in f.read().splitlines()]
    else:
        test_files = sum([dataset_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])

    print('{} files to test'.format(len(test_files)))

    for file in tqdm(test_files):

        img = imread(file).astype(np.float32)

        h,w,_ = img.shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))

        tensor_img = torch.from_numpy(img).unsqueeze(0)
        if args.original_input:
            tensor_img = (tensor_img).to(device)
        else:
            tensor_img = ((tensor_img/255 - 0.5)/0.2).to(device)

        handles = []
        for name, module in disp_net.named_modules():
            if name in fix_info.keys():
                handle = module.register_forward_hook(quantHookWithName(name))
                handles.append(handle)
        output = disp_net(tensor_img)[0]
        for handle in handles:
            handle.remove()

        if args.output_disp:
            disp = (255*tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8)
            imsave(output_dir/'{}_disp{}'.format(file.namebase,file.ext), disp)
        if args.output_depth:
            depth = 1/output
            depth = (255*tensor2array(depth, max_value=10, colormap='rainbow')).astype(np.uint8)
            imsave(output_dir/'{}_depth{}'.format(file.namebase,file.ext), depth)
    
    if not args.quantize_weights:
        return
    with open(output_dir/'dispnet_fix_info.txt', 'w') as f:
        count = 0
        for key, value in fix_info.items():
            f.write('{count} {layer} 8 {input} 8 {output} 8 {weight} 8 {bias} \n'.format(
                    count=count, layer=key, input=int(value['input']), output=int(value['output']), weight=int(value['weight']), bias=int(value['bias'])))
            count += 1

if __name__ == '__main__':
    main()
