#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo
from torch.nn import Parameter
import os

class INCEPTION_V3(nn.Module):
    def __init__(self):
        super(INCEPTION_V3, self).__init__()
        self.model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        state_dict = model_zoo.load_url(url, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        for param in self.model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
    def forward(self, input):
        x = input * 0.5 + 0.5  # [-1.0, 1.0] --> [0, 1.0]
        x[:, 0] = (x[:, 0] - 0.485) / 0.229     # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        x[:, 1] = (x[:, 1] - 0.456) / 0.224     # --> mean = 0, std = 1
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)  # --> fixed-size input: batch x 3 x 299 x 299
        x = self.model(x)  # 299 x 299 x 3
        x = nn.Softmax()(x)
        return x

class Self_Attn(nn.Module):
    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1)) 
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X N X C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # B X N X N
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out, attention

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()
    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))
        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))
    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False
    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=True)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=True)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)
    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])

def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)

def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block

def Block3x3_relu(in_planes, out_planes):  # Keep the spatial size
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out

class CA_NET(nn.Module):
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.DIMENSION
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.fc = nn.Linear(self.t_dim, self.ef_dim * 4, bias=True)
        self.relu = GLU()
    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.ef_dim]
        logvar = x[:, self.ef_dim:]
        return mu, logvar
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar

# ############## G networks ################################################
class INIT_STAGE_G(nn.Module):   #stage_G_1
    def __init__(self, ngf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf  # gf_dim=64
        if cfg.GAN.B_CONDITION:   #true
            self.in_dim = cfg.GAN.Z_DIM + cfg.GAN.EMBEDDING_DIM  #cfg.GAN.Z_DIM=100 cfg.GAN.EMBEDDING_DIM=128
        else:
            self.in_dim = cfg.GAN.Z_DIM
        self.define_module()
    def define_module(self):
        in_dim = self.in_dim  #228
        ngf = self.gf_dim   #64
        self.fc = nn.Sequential( 
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),   #in_dim=228 ngf=64
            nn.BatchNorm1d(ngf * 4 * 4 * 2),      #ngf=64
            GLU())   #64*4*4*1
        self.upsample1 = upBlock(ngf, ngf // 2)   #32*8*8*1
        self.upsample2 = upBlock(ngf // 2, ngf // 4)  #16*16*16*1
        self.upsample3 = upBlock(ngf // 4, ngf // 8)  #8*32*32*1
        self.upsample4 = upBlock(ngf // 8, ngf // 16)  #4*64*64*1
        self.atten_64 = Self_Attn(64, 'relu')
    def forward(self, z_code, c_code=None):
        if cfg.GAN.B_CONDITION and c_code is not None:
            in_code = torch.cat((c_code, z_code), 1)  #[1,228]
        else:
            in_code = z_code
        out_code = self.fc(in_code)  #64*4*4
        out_code = out_code.view(-1, self.gf_dim, 4, 4) # state_size 1 x (16x64) x 4 x 4
        out_code = self.upsample1(out_code)  # state_size 1 x (8x64) x 8 x 8
        out_code = self.upsample2(out_code)  # state_size 1 x (4x64) x 16 x 16
        out_code = self.upsample3(out_code)  # state_size 1 x (2x64) x 32 x 32
        out_code = self.upsample4(out_code)  # state_size 1 x (1x64) x 64 x 64
        out_code, attention_64 = self.atten_64(out_code)
        return out_code

class NEXT_STAGE_G(nn.Module):  #stage_G_2
    def __init__(self, ngf,num_residual=cfg.GAN.R_NUM):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf   #64
        #self.G_id = G_id
        if cfg.GAN.B_CONDITION:
            self.ef_dim = cfg.GAN.EMBEDDING_DIM  #128
        else:
            self.ef_dim = cfg.GAN.Z_DIM   #100
        self.num_residual = num_residual   #2
        self.define_module()
        self.atten_128 = Self_Attn(32, 'relu')
        #if G_id == 128:
        #    self.atten_128 = Self_Attn(32, 'relu')
        #elif G_id == 256:
        #    self.atten_256 = Self_Attn(16, 'relu')
    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)
    def define_module(self):
        ngf = self.gf_dim  #64
        efg = self.ef_dim  #128
        self.jointConv = Block3x3_relu(ngf + efg, ngf)
        self.residual = self._make_layer(ResBlock, ngf)
        self.upsample = upBlock(ngf, ngf // 2)
    def forward(self, h_code, c_code):
        s_size = h_code.size(2)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)
        h_c_code = torch.cat((c_code, h_code), 1)  # state_size (ngf+egf) x in_size x in_size
        out_code = self.jointConv(h_c_code)  # state_size ngf x in_size x in_size
        out_code = self.residual(out_code)
        out_code = self.upsample(out_code)  # state_size ngf/2 x 2in_size x 2in_size
        #out_code, attention_128 = self.atten_128(out_code)
        #if self.G_id == 128:
        #    out_code, attention_128 = self.atten_128(out_code)
        #elif G_id == 256:
        #    out_code, attention_256 = self.atten_256(out_code)
        return out_code

class GET_IMAGE_G(nn.Module):   #generate image
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh())
    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img

class GET_IMAGE_G_espcn(nn.Module):   #generate image
    def __init__(self, ngf):
        super(GET_IMAGE_G_espcn, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, ngf//2),
            nn.BatchNorm2d(ngf//2),
            nn.Tanh(),
            conv3x3(ngf//2, ngf//4),
            nn.BatchNorm2d(ngf//4),
            nn.Tanh(),
            conv3x3(ngf//4, ngf//4),
            nn.BatchNorm2d(ngf//4),
            nn.Tanh(),
            conv3x3(ngf//4, 3),
            nn.Tanh()
            )
        self.pixel_shuffle = nn.PixelShuffle(2)
    def forward(self, h_code):
        out_img = self.img(h_code)
        out_img = self.pixel_shuffle(out_img)
        return out_img

class VGG19(torch.nn.Module):
    def __init__(self,model_dir='/home/exp1/classify'):
        super(VGG19, self).__init__()
        model = models.vgg19()
        model = model.cuda()
        model.classifier._modules['6'] = nn.Linear(4096, 102)
        model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model_VGG19_flower.pth')))
        features = list(model.features)[:36]
        self.features = nn.ModuleList(features).eval()
    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 17, 26,35}:
                results.append(x)
        return results

class G_NET_64(nn.Module):
    def __init__(self):
        super(G_NET_64, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM  #self.gf_dim=64
        self.define_module()
    def define_module(self):
        if cfg.GAN.B_CONDITION:
            self.ca_net = CA_NET()
        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(self.gf_dim * 16)
            self.img_net1 = GET_IMAGE_G(self.gf_dim)      #16 times
    def forward(self, z_code, text_embedding=None):
        if cfg.GAN.B_CONDITION and text_embedding is not None:
            c_code, mu, logvar = self.ca_net(text_embedding)
        else:
            c_code, mu, logvar = z_code, None, None
        fake_imgs = []
        if cfg.TREE.BRANCH_NUM > 0:
            h_code1 = self.h_net1(z_code, c_code)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
        return fake_imgs, mu, logvar,h_code1


class G_NET_128(nn.Module):
    def __init__(self):
        super(G_NET_128, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM  # self.gf_dim=64
        self.define_module()
    def define_module(self):
        if cfg.GAN.B_CONDITION:
            self.ca_net = CA_NET()
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(self.gf_dim)
            self.img_net2 = GET_IMAGE_G(self.gf_dim // 2)
    def forward(self, z_code, h_code1, text_embedding=None):
        if cfg.GAN.B_CONDITION and text_embedding is not None:
            c_code, mu, logvar = self.ca_net(text_embedding)
        else:
            c_code, mu, logvar = z_code, None, None
        fake_imgs = []
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2 = self.h_net2(h_code1, c_code)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
        return fake_imgs, mu, logvar , h_code2

class G_NET_256(nn.Module):
    def __init__(self):
        super(G_NET_256, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM  # self.gf_dim=64
        self.define_module()
    def define_module(self):
        if cfg.GAN.B_CONDITION:
            self.ca_net = CA_NET()
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net2 = NEXT_STAGE_G(self.gf_dim // 2)
            self.img_net2 = GET_IMAGE_G(self.gf_dim // 4)
    def forward(self, z_code, h_code2, text_embedding=None):
        if cfg.GAN.B_CONDITION and text_embedding is not None:
            c_code, mu, logvar = self.ca_net(text_embedding)
        else:
            c_code, mu, logvar = z_code, None, None
        fake_imgs = []
        if cfg.TREE.BRANCH_NUM > 1:
            h_code3 = self.h_net2(h_code2, c_code)
            fake_img3 = self.img_net2(h_code3)
            fake_imgs.append(fake_img3)
        return fake_imgs, mu, logvar, h_code3

# ############## D networks ################################################
def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True))
    return block

def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True))
    return block

def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False), # --> state_size  ndf x in_size/2 x in_size/2
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # --> state size 2ndf x x in_size/4 x in_size/4
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), # --> state size 4ndf x in_size/8 x in_size/8
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # --> state size 8ndf x in_size/16 x in_size/16
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True))
    return encode_img

class D_NET64(nn.Module):  # For 64 x 64 images
    def __init__(self):
        super(D_NET64, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()
    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.spectral = SpectralNorm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4))
        self.sigmoid = nn.Sigmoid()
        self.logits = nn.Sequential(
           nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
           nn.Sigmoid())
        self.compute_label = nn.Conv2d(ndf * 8, 200, kernel_size=4, stride=4)
        if cfg.GAN.B_CONDITION:   #cfg.GAN.B_CONDITION=True
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            self.uncond_logits = nn.Sequential(
                 nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                 nn.Sigmoid())
            # self.spectral_1 = SpectralNorm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4))
            # self.sigmoid_1 = nn.Sigmoid()
    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            h_c_code = torch.cat((c_code, x_code), 1)  # state size (ngf+egf) x 4 x 4
            h_c_code = self.jointConv(h_c_code)  # state_size ngf x in_size x in_size
        else:
            h_c_code = x_code
        output = self.logits(h_c_code)
        label_value = self.compute_label(h_c_code)
        # output = self.spectral(h_c_code)
        # output = self.sigmoid(output)
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            # out_un = self.spectral_1(x_code)
            # out_uncond = self.sigmoid_1(out_un)
            return [output.view(-1), out_uncond.view(-1),label_value]
        else:
            return [output.view(-1)]

class D_NET128(nn.Module):  # For 128 x 128 images
    def __init__(self):
        super(D_NET128, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()
    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)
        self.compute_label = nn.Conv2d(ndf * 8, 200, kernel_size=4, stride=4)
        self.spectral = SpectralNorm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4))
        self.sigmoid = nn.Sigmoid()
        # self.logits = nn.Sequential(
        #    nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
        #    nn.Sigmoid())
        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            self.spectral_1 = SpectralNorm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4))
            self.sigmoid_1 = nn.Sigmoid()
            # self.uncond_logits = nn.Sequential(
            # nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            # nn.Sigmoid())
    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s32_1(x_code)
        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            h_c_code = torch.cat((c_code, x_code), 1)  # state_size (ngf+egf) x 4 x 4
            h_c_code = self.jointConv(h_c_code)  # state_size ngf x in_size x in_size
        else:
            h_c_code = x_code
        output = self.spectral(h_c_code)
        output = self.sigmoid(output)
        #output = self.logits(h_c_code)
        label_value = self.compute_label(h_c_code)
        if cfg.GAN.B_CONDITION:
            #out_uncond = self.uncond_logits(x_code)
            output_1 = self.spectral_1(h_c_code)
            out_uncond = self.sigmoid_1(output_1)
            return [output.view(-1), out_uncond.view(-1), label_value]
        else:
            return [output.view(-1)]

class D_NET256(nn.Module):  # For 256 x 256 images
    def __init__(self):
        super(D_NET256, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM    #64
        self.ef_dim = cfg.GAN.EMBEDDING_DIM   #128
        self.define_module()
    def define_module(self):
        ndf = self.df_dim   #64
        efg = self.ef_dim   #128
        self.img_code_s16 = encode_image_by_16times(ndf)   #8ndf x 16 x 16
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)   #16ndf x 8 x 8
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)  #32ndf x 4 x 4
        self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)  #16ndf x 4 x 4
        self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)   #8ndf x 4 x 4
        self.compute_label = nn.Conv2d(ndf * 8, 200, kernel_size=4, stride=4)
        self.spectral = SpectralNorm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4))
        self.sigmoid = nn.Sigmoid()
        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())
        if cfg.GAN.B_CONDITION:   #True
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            self.spectral_1 = SpectralNorm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4))
            self.sigmoid_1  = nn.Sigmoid()
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())
    def forward(self, x_var, c_code=None):
        self.perceptual_loss = []
        x_code = self.img_code_s16(x_var)
        #self.perceptual_loss.append([Variable(x_code),[8*64,256]])
        x_code = self.img_code_s32(x_code)
        #self.perceptual_loss.append([Variable(x_code),[16*64,64]])
        x_code = self.img_code_s64(x_code)
        #self.perceptual_loss.append([Variable(x_code),[32*64,16]])
        x_code = self.img_code_s64_1(x_code)
        #self.perceptual_loss.append([x_code,[16*64,16]])
        x_code = self.img_code_s64_2(x_code)
        #self.perceptual_loss.append(x_code)
        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            h_c_code = torch.cat((c_code, x_code), 1) # state_size (ngf*8+egf) x 4 x 4
            h_c_code = self.jointConv(h_c_code)  # state_size ngf*8 x 4 x 4
            #self.perceptual_loss.append([h_c_code,[8*64,16]])
        else:
            h_c_code = x_code
        # output = self.spectral(h_c_code)
        # output = self.sigmoid(output)
        output = self.logits(h_c_code)
        label_value = self.compute_label(h_c_code)
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            # output_1 = self.spectral_1(h_c_code)
            # out_uncond = self.sigmoid_1(output_1)
            return [output.view(-1), out_uncond.view(-1), label_value]
        else:
            return [output.view(-1)]