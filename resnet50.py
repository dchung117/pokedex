import torch
from torch import nn, flatten, optim
from torch.nn import Conv2d, BatchNorm2d, ReLU, Sequential, MaxPool2d, AdaptiveAvgPool2d, Linear
from torch.nn.init import kaiming_normal_, constant_
import torchvision.models as models

##  BOTTLENECK STRUCTURE ##
def check_conv_params(conv_params):
    """Checks the argument 'conv_params' for errors."""
    assert(isinstance(conv_params, dict)), conv_params + ' must be a dictionary containing parameters.'
    params_req = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'bias']
    assert(all([p in params_req for p in conv_params])), 'Need to specify ' + ', '.join(params_req) + '.'

def check_norm_feats(norm_feats):
    """Checks the argument 'norm_feats' for errors."""
    assert(isinstance(norm_feats, int)), norm_feats + ' must be an integer specifying the number of features (channels) for batch normalization.'
    assert(norm_feats > 0), norm_feats + ' must be a positive integer.'

class Bottleneck(nn.Module):

    expansion = 4
    def __init__(self, conv1_params, norm1_feats, conv2_params, norm2_feats, conv3_params, norm3_feats, act_func=ReLU(inplace=True), 
                dsmp=False, dsmp_conv_params=None, dsmp_norm_feats=None):
        super(Bottleneck, self).__init__()

        #  Run error checks on all arguments
        module_args = locals().copy()
        for arg in module_args:
            if arg[0:4] == 'conv':
                check_conv_params(module_args[arg])
            elif arg[0:4] == 'norm':
                check_norm_feats(module_args[arg])
            elif arg[0:4] == 'dsmp':
                if module_args[arg] is None:
                    continue
                elif 'conv' in arg:
                    check_conv_params(module_args[arg])
                elif 'norm' in arg:
                    check_norm_feats(module_args[arg])
                else: #  check 'dsmp' boolean argument
                    assert(isinstance(module_args[arg], bool)), arg + ' must be a boolean.'

        #  Compression Layer (Dimensionality Reduction)
        self.conv1_params = conv1_params
        self.conv1 = Conv2d(**self.conv1_params)
        self.norm1_feats = norm1_feats
        self.norm1 = BatchNorm2d(self.norm1_feats)

        #  Downsampling Layer (i.e. learn localized patterns and shrink feature maps)
        self.conv2_params = conv2_params
        self.conv2 = Conv2d(**self.conv2_params)
        self.norm2_feats = norm2_feats
        self.norm2 = BatchNorm2d(self.norm2_feats)

        #  Expansion Layer (Dimensionality Increase)
        self.conv3_params = conv3_params
        self.conv3 = Conv2d(**self.conv3_params)
        self.norm3_feats = norm3_feats
        self.norm3 = BatchNorm2d(self.norm3_feats)

        #  Additional Downsampling Layer (i.e. processing layer to match number of feature maps and feature map sizes to bottleneck output)
        self.dsmp = dsmp
        if self.dsmp:
            self.dsmp_conv_params = dsmp_conv_params
            self.dsmp_conv = Conv2d(**self.dsmp_conv_params)
            self.dsmp_norm_feats = dsmp_norm_feats
            self.dsmp_norm = BatchNorm2d(self.dsmp_norm_feats)

        #  Activation function
        self.act_func = act_func

    def forward(self, x):
        #  Store input tensor
        identity = x

        #  Compression Layer
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act_func(out)

        #  Downsampling Layer
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act_func(out)

        #  Expansion Layer
        out = self.conv3(out)
        out = self.norm3(out)

        #  Additional Downsampling Layer (if applicable)
        if self.dsmp:
            identity = self.dsmp_conv(x)
            identity = self.dsmp_norm(identity)

        #  Residual Summation
        out += identity

        # Apply activation function
        out = self.act_func(out)

        return out

## RESNET50 MODEL ##
class ResNet50(nn.Module):
    def __init__(self, num_classes, layer1, layer2, layer3, layer4, pretrained=True, zero_init_residuals=False, verbose=True):
        super(ResNet50, self).__init__()

        #  Pre-processing (downsample original image size and expand dimensions)
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2,  padding=3, bias=False) # downsample by 1/2
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1) #  downsample by 1/2

        #  Four Residual Layers - each layer contains series of bottleneck layers
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4

        #  Global Average Pooling - compute averages for each feature map after final residual layer
        self.avgpool = AdaptiveAvgPool2d(output_size=(1, 1))

        #  Fully Connected Layer - used to compute class probabilities
        self.fc = Linear(512 * 4, num_classes)

        #  Flag to load pretrained weights
        self.pretrained = pretrained
        self.verbose = verbose

        def __print_status(*msg):
            if self.verbose:
                print(*msg)

        #  Load ResNet50 state_dict and update state_dict
        if self.pretrained:
            __print_status('Loading ImageNet weights of ResNet50...')
            res50 = models.resnet50(pretrained=True)
            res50_state_dict = {}
            for p, q in zip(self.state_dict(), res50.state_dict()):
                if 'fc' in p and 'fc' in q:
                    __print_status('Randomly initializing fully connected layers', p, q)
                    if 'weight' in p and 'weight' in q:
                        res50_state_dict[p] = torch.rand(num_classes, 512 * 4, dtype=torch.float, requires_grad=True)
                    elif 'bias' in p and 'bias' in q:
                        res50_state_dict[p] = torch.rand(num_classes, dtype=torch.float, requires_grad=True)
                elif p == q:
                    res50_state_dict[p] = res50.state_dict()[q]
                elif 'norm' in p.split('.')[2] and 'bn' in q.split('.')[2]:
                    __print_status('Batch norm ', p, q)
                    res50_state_dict[p] = res50.state_dict()[q]
                elif p.split('.')[2] == 'dsmp_conv' and q.split('.')[2:4] == ['downsample', '0']:
                    __print_status('Downsample convolution ', p, q)
                    res50_state_dict[p] = res50.state_dict()[q]
                elif p.split('.')[2] == 'dsmp_norm' and q.split('.')[2:4] in [['downsample', '0'], ['downsample', '1']]:
                    __print_status('Downsample batch norm ', p, q)
                    res50_state_dict[p] = res50.state_dict()[q]
                else:
                    raise NameError(p, q, ' are incompatible.')
            self.load_state_dict(res50_state_dict)
            print()

            #  Set all parameters (besides fully connected layer) to be untrainable
            __print_status('Only training fully connected layer...')
            batchnorm_strs = ['running_mean', 'running_var', 'num_batches_tracked']
            param_state_dict = [x_name for x_name in self.state_dict() if not any([bn_str in x_name for bn_str in batchnorm_strs])]
            for param, param_name in zip(self.parameters(), param_state_dict):
                if 'fc' in param_name:
                    __print_status(param_name, 'requires gradient.')
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        #  Initialize model weights before training entire ResNet50 model
        else:
            __print_status('Initializing layer weights and biases...')
            for m in self.modules():
                if isinstance(m, Conv2d): #  Normalize convolutional layer weights
                    kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, BatchNorm2d): #  Initialize batch normalization weights and biases
                    constant_(m.weight, 1)
                    constant_(m.bias, 0)

            #  Initiailize last batch normalization weights to 0 (optionally convert residual layer into an identity function -> add input to output)
            if zero_init_residuals:
                __print_status('Zero-initializing weights of last batch normalization layer for each bottleneck module...')
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_(m.norm3.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = flatten(x, 1)
        x = self.fc(x)

        return x