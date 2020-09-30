import torch
from resnet50 import Bottleneck
from copy import deepcopy
from torch.nn import Sequential

def make_conv_dict(in_chls, out_chls, k_sz, stride, pad, bias):
    """Returns a dictionary of arguments for the Conv2d blocks."""
    return {'in_channels': in_chls, 'out_channels': out_chls, 'kernel_size': k_sz, 'stride': stride, 'padding': pad, 'bias': bias}

def build_res_layers(layer_num, in_chls=64):
    #  Check for errors in arguments
    assert(layer_num in [1, 2, 3, 4]), 'Only four residual layers in ResNet50 model architecture.'
    assert(in_chls >= 64), 'Number of input channels after image pre-processing must be at least 64.'

    #  Construct bottleneck layer w/ downsampling
    dim_red_sz = int(max(in_chls/2, 64)) #  Get num of dimensions after dimensionality reduction
    dim_exp_sz = dim_red_sz * 4 #  Get num of dimensions after dimensionality expansion

    if layer_num == 1:
        ds_stride = 1 #  No downsampling for residual layer 1
    else:
        ds_stride = 2

    ds_conv1_params = make_conv_dict(in_chls=in_chls, out_chls=dim_red_sz, k_sz=1, stride=1, pad=0, bias=False) #  dimensionality reduction
    ds_conv2_params = make_conv_dict(in_chls=dim_red_sz, out_chls=dim_red_sz, k_sz=3, stride=ds_stride, pad=1, bias=False) #  downsampling
    ds_conv3_params = make_conv_dict(in_chls=dim_red_sz, out_chls=dim_exp_sz, k_sz=1, stride=1, pad=0, bias=False) #  dimensionality expansion
    ds_conv_params = make_conv_dict(in_chls=in_chls, out_chls=dim_exp_sz, k_sz=1, stride=ds_stride, pad=0, bias=False) #  input downsampling

    ds_bottle = Bottleneck(conv1_params=ds_conv1_params, norm1_feats=dim_red_sz, conv2_params=ds_conv2_params, norm2_feats=dim_red_sz, conv3_params=ds_conv3_params, norm3_feats=dim_exp_sz, 
                        dsmp=True, dsmp_conv_params=ds_conv_params, dsmp_norm_feats=dim_exp_sz)

    #  Construct bottleneck layer w/o downsampling
    conv1_params = make_conv_dict(in_chls=dim_exp_sz, out_chls=dim_red_sz, k_sz=1, stride=1, pad=0, bias=False)
    conv2_params = make_conv_dict(in_chls=dim_red_sz, out_chls=dim_red_sz, k_sz=3, stride=1, pad=1, bias=False)
    conv3_params = make_conv_dict(in_chls=dim_red_sz, out_chls=dim_exp_sz, k_sz=1, stride=1, pad=0, bias=False)

    bottle = Bottleneck(conv1_params=conv1_params, norm1_feats=dim_red_sz, conv2_params=conv2_params, norm2_feats=dim_red_sz, conv3_params=conv3_params, norm3_feats=dim_exp_sz)

    #  Return a sequence of modules (depending on layer number)
    if layer_num in [1, 4]: #  2 standard bottlenecks
        bottle_list = [ds_bottle] + list(map(deepcopy, [bottle]*2))
        return Sequential(*bottle_list)
    elif layer_num == 2: #  3 standard bottlenecks
        bottle_list = [ds_bottle] + list(map(deepcopy, [bottle]*3))
        return Sequential(*bottle_list)
    elif layer_num == 3: #  5 standard bottlenecks
        bottle_list = [ds_bottle] + list(map(deepcopy, [bottle]*5))
        return Sequential(*bottle_list)

def get_device(verbose=False):
    """Returns the current device (CUDA if available, else CPU)."""
    current_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if verbose:
        print('Current device: ', current_device)
    return current_device