import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import kornia
import torch.nn.init as init


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
    
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class ResidualBlock(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.bn = nn.BatchNorm2d(nf)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn(self.conv1(x)), inplace=True)
        out = self.conv2(out)
        return identity + out

###########################################################################################################


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SGE(nn.Module):
    def __init__(self, dw_channel):
        super().__init__() 
        self.dwc = nn.Conv2d(in_channels=dw_channel //2, out_channels=dw_channel//2, kernel_size=3, padding=1, stride=1, groups=dw_channel//2, bias=True)
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwc(x1)
        return x1 * x2
    
class SpaBlock(nn.Module):
    def __init__(self, nc, DW_Expand = 2,  FFN_Expand=2, drop_out_rate=0.):
        super(SpaBlock, self).__init__()
        dw_channel = nc * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=nc, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True) # the dconv
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=nc, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * nc
        self.conv4 = nn.Conv2d(in_channels=nc, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=nc, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(nc)
        self.norm2 = LayerNorm2d(nc)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, nc, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, nc, 1, 1)), requires_grad=True)

    def forward(self, x):

        x = self.norm1(x) # size [B, C, H, W]

        x = self.conv1(x) # size [B, 2*C, H, W]
        x = self.conv2(x) # size [B, 2*C, H, W]
        x = self.sg(x)    # size [B, C, H, W]
        x = x * self.sca(x) # size [B, C, H, W]
        x = self.conv3(x) # size [B, C, H, W]

        x = self.dropout1(x)

        y = x + x * self.beta # size [B, C, H, W]

        x = self.conv4(self.norm2(y)) # size [B, 2*C, H, W]
        x = self.sg(x)  # size [B, C, H, W]
        x = self.conv5(x) # size [B, C, H, W]

        x = self.dropout2(x)

        return y + x * self.gamma

class FreBlock(nn.Module):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.fpre = nn.Conv2d(nc, nc, 1, 1, 0)
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process2 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(self.fpre(x), norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        pha = self.process2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')

        return x_out+x

    
class SFBlock(nn.Module):
    def __init__(self, nc, DW_Expand = 2,  FFN_Expand=2):
        super(SFBlock, self).__init__()
        dw_channel = nc * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=nc, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True) # the dconv
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=nc, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        self.fatt = FreBlock(dw_channel // 2)
        self.sge = SGE(dw_channel)

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * nc
        self.conv4 = nn.Conv2d(in_channels=nc, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=nc, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(nc)
        self.norm2 = LayerNorm2d(nc)

        self.beta = nn.Parameter(torch.zeros((1, nc, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, nc, 1, 1)), requires_grad=True)

    def forward(self, x):

        x = self.norm1(x) # size [B, C, H, W]

        x = self.conv1(x) # size [B, 2*C, H, W]
        x = self.conv2(x) # size [B, 2*C, H, W]
        x = self.sge(x)    # size [B, C, H, W]
      
        x = self.fatt(x)
        x = self.conv3(x) # size [B, C, H, W]

        y = x + x * self.beta # size [B, C, H, W]

        x = self.conv4(self.norm2(y)) # size [B, 2*C, H, W]
        x = self.sg(x)  # size [B, C, H, W]
        x = self.conv5(x) # size [B, C, H, W]

        return y + x * self.gamma
    
class ProcessBlock(nn.Module):
    def __init__(self, in_nc, spatial = True):
        super(ProcessBlock,self).__init__()
        self.spatial = spatial
        self.spatial_process = SpaBlock(in_nc) if spatial else nn.Identity()
        self.frequency_process = FreBlock(in_nc)
        self.cat = nn.Conv2d(2*in_nc,in_nc,1,1,0) if spatial else nn.Conv2d(in_nc,in_nc,1,1,0)

    def forward(self, x):
        xori = x
        x_freq = self.frequency_process(x)
        x_spatial = self.spatial_process(x)
        xcat = torch.cat([x_spatial,x_freq],1)
        x_out = self.cat(xcat) if self.spatial else self.cat(x_freq)

        return x_out+xori

class SFNet(nn.Module):

    def __init__(self, nc,n=5):
        super(SFNet,self).__init__()

        self.list_block = list()
        for index in range(n):

            self.list_block.append(ProcessBlock(nc,spatial=False))
  
        self.block = nn.Sequential(*self.list_block)

    def forward(self, x):

        x_ori = x
        x_out = self.block(x_ori)
        xout = x_ori + x_out

        return xout

class AmplitudeNet_skip(nn.Module):
    def __init__(self, nc,n=1):
        super(AmplitudeNet_skip,self).__init__()
        
        self.conv_init = nn.Conv2d(3, nc, 1, 1, 0)
        self.conv1 = SFBlock (nc)
        self.conv2 = SFBlock (nc)
        self.conv3 = SFBlock (nc)
        self.conv_out = nn.Conv2d(nc, 3, 1, 1, 0)

    def forward(self, x):
        
        x_lr = F.interpolate(x, scale_factor=0.5, mode='bilinear') # Resize and Normalize SNR map
        
        x_lr = self.conv_init(x_lr)
        x_lr = self.conv1(x_lr)
        x_lr = self.conv2(x_lr)
        x_lr = self.conv3(x_lr)
        x_lr = self.conv_out(x_lr)
        
        xout = F.interpolate(x_lr, scale_factor=2, mode='bilinear') # Resize and Normalize SNR map
        
        return xout

    
###########################################################################################################

class SG(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    

class SGE(nn.Module):
    def __init__(self, dw_channel):
        super().__init__() 
        self.dwc = nn.Conv2d(in_channels=dw_channel //2, out_channels=dw_channel//2, kernel_size=3, padding=1, stride=1, groups=dw_channel//2, bias=True)
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwc(x1)
        return x1 * x2

class FLOL(nn.Module):
    def __init__(self, nf=16):
        super(FLOL, self).__init__()

        # AMPLITUDE ENHANCEMENT
        # 幅度条件子网络

        self.AmpNet = nn.Sequential(
            AmplitudeNet_skip(8),
            nn.Sigmoid()
        )

        self.nf = nf

        # 定义残差块模板
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)

        # 空域编码器：逐步下采样（stride=2），通道数从6→nf→nf→nf   [B, 6, H, W]->[B, nf, H/4, W/4]
        self.conv_first_1 = nn.Conv2d(3 * 2, nf, 3, 1, 1, bias=True)
        self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)

        # 1个残差块提取基础特征
        self.feature_extraction = make_layer(ResidualBlock_noBN_f, 1)

        # 1个残差块提取重构特征
        self.recon_trunk = make_layer(ResidualBlock_noBN_f, 1)

        # 解码器：上采样+拼接
        self.upconv1 = nn.Conv2d(nf*2, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf*2, nf * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.transformer = SFNet(nf, n = 4)
        self.recon_trunk_light = make_layer(ResidualBlock_noBN_f, 6)

    def get_mask(self,dark):   # SNR map

        light = kornia.filters.gaussian_blur2d(dark, (5, 5), (1.5, 1.5))
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = torch.abs(dark - light)

        mask = torch.div(light, noise + 0.0001)

        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        mask = mask * 1.0 / (mask_max + 0.0001)

        mask = torch.clamp(mask, min=0, max=1.0)
        return mask.float()

    def forward(self, x, side=False):

        # AMPLITUDE ENHANCEMENT
        #--------------------------------------------------------Frequency Stage---------------------------------------------------
        _, _, H, W = x.shape
        image_fft = torch.fft.fft2(x, norm='backward')

        # 幅度谱
        mag_image = torch.abs(image_fft)

        # 相位谱
        pha_image = torch.angle(image_fft)
        
        # 幅度增益曲线
        curve_amps = self.AmpNet(x)
        
        # 调整幅度后进行傅里叶逆变换，得到频域增强后的图像
        mag_image = mag_image / (curve_amps + 0.00000001)  # * d4
        real_image_enhanced = mag_image * torch.cos(pha_image)
        imag_image_enhanced = mag_image * torch.sin(pha_image)
        img_amp_enhanced = torch.fft.ifft2(torch.complex(real_image_enhanced, imag_image_enhanced), s=(H, W),
                                           norm='backward').real

        x_center = img_amp_enhanced


        # 补边操作
        rate = 2 ** 3
        pad_h = (rate - H % rate) % rate
        pad_w = (rate - W % rate) % rate
        if pad_h != 0 or pad_w != 0:
            x_center = F.pad(x_center, (0, pad_w, 0, pad_h), "reflect")
            x = F.pad(x, (0, pad_w, 0, pad_h), "reflect")

        #------------------------------------------Spatial Stage---------------------------------------------------------------------

        
        L1_fea_1 = self.lrelu(self.conv_first_1(torch.cat((x_center,x),dim=1)))     # (B,nf,H,W)
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))   # Encoder              # (B,nf,H/2,W/2) 下采样×2
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))                          # (B,nf,H/4,W/4) 下采样×4

        # 基础特征提取
        fea = self.feature_extraction(L1_fea_3)                                    # (B,nf,H/4,W/4) 基础特征
        fea_light = self.recon_trunk_light(fea)                                    # (B,nf,H/4,W/4) 细节增强特征（6个残差块，更关注细节）

        h_feature = fea.shape[2]
        w_feature = fea.shape[3]
        mask_image = self.get_mask(x_center) # SNR Map
        mask = F.interpolate(mask_image, size=[h_feature, w_feature], mode='nearest') # Resize and Normalize SNR map

        fea_unfold = self.transformer(fea)

        channel = fea.shape[1]
        mask = mask.repeat(1, channel, 1, 1)
        fea = fea_unfold * (1 - mask) + fea_light * mask  # SNR-based Interaction  # (B,nf,H/4,W/4)

        out_noise = self.recon_trunk(fea)
        out_noise = torch.cat([out_noise, L1_fea_3], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_2], dim=1)                   # Decoder
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_1], dim=1)
        out_noise = self.lrelu(self.HRconv(out_noise))
        out_noise = self.conv_last(out_noise)
        out_noise = out_noise + x
        out_noise = out_noise[:, :, :H, :W]
        
        if side:
            return out_noise, x_center #, mag_image, x_center, mask_image
        else:
             return out_noise

        
##############################################################################

def create_model():
    
    net = FLOL(nf=16)
    return net