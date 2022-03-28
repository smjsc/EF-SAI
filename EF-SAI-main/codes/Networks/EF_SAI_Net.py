import torch
import torch.nn as nn
import torch.nn.functional as F
from codes.Networks.submodules import define_G,  ChannelAttentionv2, PatchEmbed, PatchUnEmbed, FusionSwinTransformerBlock
from timm.models.layers import  trunc_normal_
# cofiguration for convolutional layers
cfg_cnn = [(30*2, 8, 1, 1, 3),
           (8+60, 16, 1, 2, 5),
           (16+60, 32, 1, 3, 7)]

cfg_cnn2 = [(30, 8, 1, 1, 3),
           (8+30, 16, 1, 2, 5),
           (16+30, 32, 1, 3, 7)]

cfg_cnn3 = [(30, 8, 1, 1, 3),
           (8+30, 16, 1, 2, 5),
           (16+30, 32, 1, 3, 7)]

cfg_cnn4 = [(1, 8, 1, 1, 3),
           (8+1, 16, 1, 2, 5),
           (1+16, 32, 1, 3, 7)]

# cofiguration for spiking layers
cfg_snn =  [(2, 8, 1, 0, 1),
           (8, 16, 1, 1, 3),
           (16+8, 32, 1, 3, 7)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 0.5 # neuronal threshold
lens = 0.5 # hyper-parameters of approximate function
decay = 0.95 # decay constants

# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()  ## potential reset here

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply
# membrane potential update

def mem_update(ops, x, mem, spike):
    mem = mem * decay * (1. - spike) + ops(x)
    spike = act_fun(mem) # act_fun : approximation firing function
    return mem, spike
class EF_SAI_Net(nn.Module):
    def __init__(self):
        super(EF_SAI_Net, self).__init__()

        ## pureCNN
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conve1=nn.Conv2d(in_channels=in_planes,out_channels=out_planes,kernel_size=kernel_size,stride=stride,padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn2[0]
        self.convf1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                                padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn3[0]
        self.convfe1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                                padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn4[0]
        self.convue1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                                padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conve2=nn.Conv2d(in_channels=in_planes,out_channels=out_planes,kernel_size=kernel_size,stride=stride,padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn2[1]
        self.convf2 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                                padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn3[1]
        self.convfe2 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                                padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn4[1]
        self.convue2 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                                padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[2]
        self.conve3=nn.Conv2d(in_channels=in_planes,out_channels=out_planes,kernel_size=kernel_size,stride=stride,padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn2[2]
        self.convf3 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                                padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn3[2]
        self.convfe3 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                                padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn4[2]
        self.convue3 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                                padding=padding)
        ## Define SNN encoder
        in_planes, out_planes, stride, padding, kernel_size = cfg_snn[0]
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                               padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_snn[1]
        self.conv2 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                               padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_snn[2]
        self.conv3 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                               padding=padding)

        ##################################################################
        ####################### Swin Config ##############################
        ##################################################################
        window_size = 8
        self.window_size = window_size
        self.d = 6
        mlp_ratio = 2.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        ape = False
        self.ape = ape
        patch_norm = True
        self.patch_norm = patch_norm
        #self.swintf = swintf(img_size=(256, 256), patch_size= 16, window_size=8, img_range=1., depths=[4, 4], embed_dim=64*3, num_heads=[4, 4], mlp_ratio=2)
        img_size = (256, 256)
        patch_size = 16
        embed_dim = 32
        self.num_features = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.d)]  # stochastic depth decay rule

        self.swin_blocks = nn.ModuleList([
            FusionSwinTransformerBlock(dim=embed_dim, input_resolution=(patches_resolution[0], patches_resolution[1]),
                                       num_heads=4, window_size=8,
                                       shift_size=0 if (i % 2 == 0) else window_size // 2,
                                       mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       drop=drop_rate, attn_drop=attn_drop_rate,
                                       drop_path=dpr[i],
                                       norm_layer=norm_layer)
            for i in range(self.d)])

        self.norm_eoa = norm_layer(self.num_features)
        self.norm_f = norm_layer(self.num_features)
        self.norm_eaa = norm_layer(self.num_features)


        self.ca1 = ChannelAttentionv2(out_planes, 30)
        #self.sa1 = SpatialAttentionv3()
        # self.ca2 = ChannelAttention(out_planes + 30)
        # self.sa2 = SpatialAttention()
        # self.ca3 = ChannelAttention(out_planes + 30)
        # self.sa3 = SpatialAttention()
        # self.unet = UNet(60, 1)
        # self.unet_e1 = UNet(60, 1)
        self.Gen = define_G(32 * 3, 1, 64, 'resnet_9blocks',norm='batch', use_dropout=False)  # dropout
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    def forward(self, inpute, inputf, inputfe, time_window = 20):
        batch_size = inpute.shape[0]
        inpsize = inpute.shape[3]
        inpe = inpute.to(device)
        inpf = inputf.to(device)
        inpfe = inputfe.to(device)
        # me = torch.mean(inpe,dim=1,keepdim=True)
        # mf = torch.mean(inpf, dim=1, keepdim=True)
        # mfe = torch.mean(inpue, dim=1, keepdim=True)


        # x1 = self.conve1(inpe)
        # x = torch.cat((x1, inpe), 1)
        # x2 = self.conve2(x)
        # x = torch.cat((x2, inpe), 1)
        # x3 = self.conve3(x)
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_snn[0][1], inpsize, inpsize, device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_snn[1][1], inpsize, inpsize, device=device)
        c3_mem = c3_spike = torch.zeros(batch_size, cfg_snn[2][1], inpsize, inpsize, device=device)
        sumspike = torch.zeros(batch_size, cfg_snn[2][1], inpsize, inpsize, device=device)

        ## SNN encoder
        for step in range(time_window):  # simulation time steps
            inp = inpe[:, step, :]
            x = inp
            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike)
            x = c1_spike
            c2_mem, c2_spike = mem_update(self.conv2, x, c2_mem, c2_spike)
            x = torch.cat((c2_spike, c1_spike), 1)
            c3_mem, c3_spike = mem_update(self.conv3, x, c3_mem, c3_spike)
            sumspike += c3_spike
            # normalize SNN output
        x3 = sumspike / time_window
        x3_shortcut = x3
        y1 = self.convf1(inpf)
        y = torch.cat((y1, inpf), 1)
        y2 = self.convf2(y)
        y = torch.cat((y2, inpf),1)
        y3 = self.convf3(y)
        y3_shortcut = y3

        z1 = self.convfe1(inpfe)
        z = torch.cat((z1, inpfe), 1)
        z2 = self.convfe2(z)
        z = torch.cat((z2, inpfe), 1)
        z3 = self.convfe3(z)
        z3_shortcut = z3
        inp_size = x3.shape[-2:]
        for i in range(self.d):
            #### fusion ####
            x3 = self.patch_embed(x3)
            y3 = self.patch_embed(y3)
            z3 = self.patch_embed(z3)
            if self.ape:
                x3 = x3 + self.absolute_pos_embed
                y3 = y3 + self.absolute_pos_embed
                z3 = z3 + self.absolute_pos_embed
            # e = self.pos_drop(e)

            x3, y3, z3 = self.swin_blocks[i](x3, y3, z3, inp_size)

            x3 = self.norm_eoa(x3)  # B L C
            x3 = self.patch_unembed(x3, inp_size)

            y3 = self.norm_f(y3)  # B L C
            y3 = self.patch_unembed(y3, inp_size)

            z3 = self.norm_eaa(z3)  # B L C
            z3 = self.patch_unembed(z3, inp_size)

        x3 = x3 + x3_shortcut
        y3 = y3 + y3_shortcut
        z3 = z3 + z3_shortcut
        inpe1 = inpe[:,:,0,:,:]
        inpe2 = inpe[:,:,0,:,:]
        inp = torch.cat((inpe1,inpe2),axis = 1)
        ca = self.ca1(x3, inp, y3, inpf, z3, inpfe)
        temp2 = torch.split(ca, 1, dim=1)
        x3 = x3*temp2[0]
        y3 = y3*temp2[1]
        z3 = z3*temp2[2]
        inp_fetures = torch.cat((x3, y3, z3), dim=1)

        outputs = self.Gen(inp_fetures)

        return outputs


if __name__ == '__main__':
    net = EF_SAI_Net()
    net = net.to(device)
    net = torch.nn.DataParallel(net, device_ids=[0])
    net.load_state_dict(torch.load('/home_ssd/LW/AIOEdata/PreTraining/PreTraining1031_total_hybrid_swinv2/Hybrid_test_stage2.pth'),strict=False)
    torch.save(net.state_dict(), './EF_SAI_Net.pth')
