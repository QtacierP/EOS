import torch
from torch import nn
import torch.nn.functional as F
import torchvision


def get_activation(activation: str):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError

def get_norm_1d(norm: str, num_features=200):
    if norm == 'bn':
        return nn.BatchNorm1d(num_features)
    elif norm == 'ln':
        return nn.LayerNorm([num_features])
    elif norm == 'in':
        return nn.InstanceNorm1d(num_features)
    else:
        raise NotImplementedError



def get_norm_2d(norm: str, num_features=200):
    if norm == 'bn':
        return nn.BatchNorm2d(num_features)
    elif norm == 'ln':
        raise NotImplementedError
    elif norm == 'in':
        return nn.InstanceNorm2d(num_features)
    else:
        raise NotImplementedError




class MLPNet(nn.Module):
    def __init__(self,  hidden_unit_list, input_dim=32*32*3, output_dim=10, activation='relu', norm='batch', init=None):
        super().__init__()
        self.layers = len(hidden_unit_list)
        fc1 = []
        use_bias = True if norm == 'bn' or norm == 'in' else False
        fc1.append(nn.Linear(input_dim, hidden_unit_list[0], bias=use_bias))
        if activation is not None:
            fc1.append(get_activation(activation))
        if norm is not None:
            fc1.append(get_norm_1d(norm, hidden_unit_list[0]))
        self.fc1 = torch.nn.Sequential(*fc1)
        for i in range(self.layers-1):
            unit  = []
            fc = nn.Linear(hidden_unit_list[i], hidden_unit_list[i+1], bias=use_bias)
            unit.append(fc)
            if activation is not None:
                unit.append(get_activation(activation))
            if norm is not None:
                unit.append(get_norm_1d(norm, hidden_unit_list[i+1]))
            setattr(self, 'fc{}'.format(i+2), torch.nn.Sequential(*unit))
        
        self.classifier = nn.Linear(hidden_unit_list[-1], output_dim)
        self.init_weights(init=init, activation=activation)
    

    def init_weights(self, init='kaiming', activation='relu'):
        if init == 'kaiming':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=activation)
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=activation)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0) 
                elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.InstanceNorm1d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        elif init == 'xavier':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0) 
                elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.InstanceNorm1d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        elif init == 'normal':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0) 
                elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.InstanceNorm1d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            return

    
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i in range(self.layers):
            x = getattr(self, 'fc{}'.format(i+1))(x)
            #x = F.relu(x)
        x = self.classifier(x)
        return x
    
class ResMLPBlock(nn.Module):
    def __init__(self, in_features, out_features, stride=1, downsample=None, activation='relu', norm='bn'):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features, bias=False)
        if activation is not None:
            self.activation1 = get_activation(activation)
        if norm is not None:
            self.norm1 = get_norm_1d(norm, out_features)
        else:
            self.norm1 = nn.Identity()
        self.fc2 = nn.Linear(out_features, out_features, bias=False)
        if activation is not None:
            self.activation2 = get_activation(activation)
        if norm is not None:
            self.norm2 = get_norm_1d(norm, out_features)
        else:
            self.norm2 = nn.Identity()
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        if hasattr(self, 'activation1'):
            out = self.activation1(out)
        if hasattr(self, 'norm1'):
            out = self.norm1(out)
        out = self.fc2(out)
        if hasattr(self, 'activation2'):
            out = self.activation2(out)
        if hasattr(self, 'norm2'):
            out = self.norm2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return out
    
class ResMLPNet(nn.Module):
    def __init__(self, hidden_unit_list, input_dim=32*32*3, output_dim=10, activation='relu', norm='batch'):
        super().__init__()
        self.layers = len(hidden_unit_list)
        fc1 = []
        use_bias = True if norm == 'bn' or norm == 'in' else False
        fc1.append(nn.Linear(input_dim, hidden_unit_list[0], bias=use_bias))
        if activation is not None:
            fc1.append(get_activation(activation))
        if norm is not None:
            fc1.append(get_norm_1d(norm, hidden_unit_list[0]))
        self.block1 = torch.nn.Sequential(*fc1)
        for i in range(self.layers-1):
            block = ResMLPBlock(hidden_unit_list[i], hidden_unit_list[i+1], stride=1, activation=activation, norm=norm)
            setattr(self, 'block{}'.format(i+2), block)
        self.classifier = nn.Linear(hidden_unit_list[-1], output_dim)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i in range(self.layers):
            x = getattr(self, 'block{}'.format(i+1))(x)
        x = self.classifier(x)
        return x




class CNNNet(nn.Module):
    def __init__(self, hidden_unit_list,  output_dim=10, activation='relu', norm='batch', num_colors=3, spatial_dim=32):
        super().__init__()
        self.layers = len(hidden_unit_list)
        conv1 = [nn.Conv2d(num_colors, hidden_unit_list[0], kernel_size=3, stride=1, padding=1)]
        if activation is not None:
            conv1.append(get_activation(activation))
        if norm is not None:
            conv1.append(get_norm_2d(norm, hidden_unit_list[0]))
        self.conv1 = torch.nn.Sequential(*conv1)
    
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        spatial_dim = spatial_dim // 2
        for i in range(self.layers-1):
            unit = []
            conv = nn.Conv2d(hidden_unit_list[i], hidden_unit_list[i+1], kernel_size=3, stride=1, padding=1)
            unit.append(conv)
            if activation is not None:
                unit.append(get_activation(activation))
            if norm is not None:
                unit.append(get_norm_2d(norm, hidden_unit_list[i+1]))
            setattr(self, 'conv{}'.format(i+2), torch.nn.Sequential(*unit))
            spatial_dim = spatial_dim // 2
        
        self.classifier = nn.Linear(hidden_unit_list[-1] * spatial_dim * spatial_dim, output_dim)
    
    def forward(self, x):
        #x = x.permute(0, 3, 1, 2)
        for i in range(self.layers):
            x = getattr(self, 'conv{}'.format(i+1))(x)
            x = self.pool(x)
        x = x.view(x.size(0), -1).contiguous()
        x = self.classifier(x)
        return x
    

class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, activation='relu', norm='batch'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        if activation is not None:
            self.activation1 = get_activation(activation)
        if norm is not None:
            self.norm1 = get_norm_2d(norm, out_channels)
        else:
            self.norm1 = nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if activation is not None:
            self.activation2 = get_activation(activation)
        if norm is not None:
            self.norm2 = get_norm_2d(norm, out_channels)
        else:
            self.norm2 = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if hasattr(self, 'activation1'):
            out = self.activation1(out)
        if hasattr(self, 'norm1'):
            out = self.norm1(out)
        out = self.conv2(out)
        if hasattr(self, 'activation2'):
            out = self.activation2(out)
        if hasattr(self, 'norm2'):
            out = self.norm2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out =out +  residual
        return out

class ResCNNNet(nn.Module):
    def __init__(self, hidden_unit_list, output_dim=10, activation='relu', norm='batch', num_colors=3, spatial_dim=32):
        super().__init__()
        self.layers = len(hidden_unit_list)
        self.conv1 = nn.Conv2d(num_colors, hidden_unit_list[0], kernel_size=3, stride=1, padding=1)
        if activation is not None:
            self.activation1 = get_activation(activation)
        if norm is not None:
            self.norm1 = get_norm_2d(norm, hidden_unit_list[0])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        spatial_dim = spatial_dim // 2
        for i in range(self.layers-1):
            block = ResConvBlock(hidden_unit_list[i], hidden_unit_list[i], stride=1, activation=activation, norm=norm)
            setattr(self, 'block{}'.format(i+1), block)
            conv = nn.Conv2d(hidden_unit_list[i], hidden_unit_list[i+1], kernel_size=3, stride=1, padding=1)
            if activation is not None:
                setattr(self, 'activation{}'.format(i+2), get_activation(activation))
            if norm is not None:
                setattr(self, 'norm{}'.format(i+2), get_norm_2d(norm, hidden_unit_list[i+1]))
            setattr(self, 'conv{}'.format(i+2), conv)
            spatial_dim = spatial_dim // 2
        self.classifier = nn.Linear(hidden_unit_list[-1] * spatial_dim * spatial_dim, output_dim)
    
    def forward(self, x):
        x = self.conv1(x)
        if hasattr(self, 'activation1'):
            x = self.activation1(x)
        if hasattr(self, 'norm1'):
            x = self.norm1(x)
        x = self.pool(x)
        for i in range(self.layers - 1):
            x = getattr(self, 'block{}'.format(i+1))(x)
            x = getattr(self, 'conv{}'.format(i+2))(x)
            x = self.pool(x)
        x = x.view(x.size(0), -1).contiguous()
        x = self.classifier(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, output_dim=10, pretrained=False):
        super().__init__()
        model = torchvision.models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, output_dim)
        self.model = model
        # reset parameters
    #     self.reset_parameters()
    #     # change pool to conv
    #     self.model.maxpool = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

    # def reset_parameters(self):
    #     for m in self.model.modules():
    #         # print classname
    #         if isinstance(m, nn.Conv2d):
    #             m.reset_parameters()
    #         if isinstance(m, nn.Linear):
    #             m.reset_parameters()
    #         if isinstance(m, (nn.BatchNorm2d)):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
            
            

    def forward(self, x):
        return self.model(x)




class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        factor = 2 if bilinear else 1
        self.up1 = (Up(256, 128 // factor, bilinear))
        self.up2 = (Up(128, 64 // factor, bilinear))
        self.up3 = (Up(64, 32 // factor, bilinear))
        self.outc = (OutConv(32, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits
    


if __name__ == '__main__':
    model = UNet()
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.shape)