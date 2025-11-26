import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dconv_model import FusedPooling, LayerNorm2d, SCAM
from unet import UNetCompress, UNetDecompress
from model_convnext import knowledge_adaptation_convnext

#그림자를 '제거한다'가 아니라 광원을 '추가한다'로 보면 어떨까?->이미지를 ;'반전'
#수정내용
#concat -> conv
#pri-norm -> peri-norm

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def dwt_haar(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)

    def forward(self, x):
        return self.dwt_haar(x)

#2x2짜리 dx, dy, dxdy(대각), 평균을 만드는건데 이걸 2x2만 하지말고 사이즈를 달리해가며 할 때 보이는 변화가 달라질 것이다. 이는 그 특이 영역 찾는거에 기초함.
#2x2를 cascading해서 하면 그게 곧 4x4 ... 이 됨. 대신 그땐 평균에 해당하는 LL만을 가지고 ㅇㅅㅇㅇ

class DWT_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = DWT()

        self.conv1x1_low = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1x1_high = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        dwt_low_frequency, dwt_high_frequency = self.dwt(x)
        dwt_low_frequency = self.conv1x1_low(dwt_low_frequency)
        dwt_high_frequency = self.conv1x1_high(dwt_high_frequency)
        return dwt_low_frequency, dwt_high_frequency

#1x1 conv아마 그냥 imbedding의 FC를 대체하는 거 아니까나

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

#채널을 기준으로 앞쪽 절반 채널과 뒤쪽 절반 채널로 쪼갠뒤 엘레와이즈 곱으로 채널을 반으로 줄이는 효과를 냄.

class CAM(nn.Module):
    def __init__(self, num_channels, compress_factor=8):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#채널 전체의 정보를 1개의 점으로 축약

        self.model = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // compress_factor, 1, padding=0, bias=True),
            nn.PReLU(),
            nn.Conv2d(num_channels // compress_factor, num_channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        #채널을 압축해서 conv후 prelu를 통해 activation을 위한 color를 학습하고->이는 즉, conv를 통해 각 채널의 조합을 통해 context(채널간의 관계)를 담는 vector로 만듦.
        #그 color vector를 다시 conv와 sigmoid로 입혀 각 채널의 중요도로 만듦.
        
        #앞의 conv의 채널을 늘리는 방향으로 시도해서 다양한 context를 고려하게 해볼까
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.model(y)
        return x * y


class DynamicDepthwiseConvolution(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, num_cnvs=4, K=2, device='cuda'):
        super(DynamicDepthwiseConvolution, self).__init__()

        self.convs = nn.ModuleList([
                        nn.Conv2d(in_ch, in_ch * K, k, stride, groups=in_ch, padding='same', padding_mode='reflect')
                            for _ in range(num_cnvs)])
#구조가 같은 conv layer를 병렬적으로 생성 dynamic하게 K, k값에 따라 채널을 조절함. 단, 학습은 독립적으로 이뤄짐
        self.weights = nn.Parameter(1 / num_cnvs * torch.ones((num_cnvs, 1), device=device, dtype=torch.float),
                                    requires_grad=True)
#각 레이어의 weight의 반영비율에 해당하는 초기값 설정 즉 가중합을 할 때 쓰는 weight->지금은 point를 전체에 묻히는데 이를 mat화?
#그럼 얘도 attention처럼 각 conv를 1point로 압축하고 그걸 conv해 만든 weight를 쓰는 attention방식 어떠냐->어차피 CAM으로 묻히는데 뭔가 <==> 이런 항아리 느낌으로 펴졌다가 합쳐지는 느낌
#지금은 local 정보 보존에 가치가 있음
        self.final_conv = nn.Conv2d(in_channels=in_ch * K, out_channels=out_ch, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        feats = 0
        for i, conv in enumerate(self.convs):
            feats += self.weights[i] * conv(x)

        return self.final_conv(feats)


class SimplifiedDepthwiseCA(nn.Module):
    def __init__(self, num_channels, k, K, device="cuda"):
        super().__init__()
        self.attention = CAM(num_channels)
        self.dc = DynamicDepthwiseConvolution(in_ch=num_channels, out_ch=num_channels, K=K, k=k, device=device)

    def forward(self, x):
        q = self.dc(x)
        w = self.attention(q)
        return torch.sigmoid(w * q)
#걍 어텐션 박기

class BlockRGB(nn.Module):
    def __init__(self, in_ch, out_ch, k_sz=3, dropout_prob=0.5,  device="cuda"):
        super(BlockRGB, self).__init__()
        self.ln = LayerNorm2d(in_ch)
        self.preln = LayerNorm2d(in_ch)#####
        self.conv1 = nn.Conv2d(in_ch, out_ch // 2, k_sz, padding=k_sz // 2, padding_mode="reflect", bias=True)
        self.postln = LayerNorm2d(out_ch // 2)#####peri-LN
        self.op1 = nn.LeakyReLU(0.2)
        self.dyndc = SimplifiedDepthwiseCA(num_channels=out_ch // 2, k=13, K=4, device=device)
        self.conv2 = nn.Conv2d(out_ch // 2, out_ch, k_sz, padding=k_sz // 2, padding_mode="reflect", bias=True)
        self.op2 = nn.LeakyReLU(0.2)

        self.rconv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch // 2, kernel_size=1, padding=0, stride=1,
                                groups=1, bias=True)
        self.rconv2 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, stride=1,
                                groups=1, bias=True)
        
        self.a1 = nn.Parameter(torch.tensor(1.0, device=device, dtype=torch.float32), requires_grad=True)
        self.a2 = nn.Parameter(torch.tensor(1.0, device=device, dtype=torch.float32), requires_grad=True)
        self.dropout1 = nn.Dropout(dropout_prob) if dropout_prob > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_prob) if dropout_prob > 0. else nn.Identity()

    def forward(self, x):
        #xf = self.ln(x)
        xf = self.preln(x)####
        #xf = self.op1(self.conv1(xf))
        xf = self.op1(self.postln(self.conv1(xf)))#####
        xf = self.dropout1(xf)
        xf += self.a1 * self.rconv1(x)#x를 1x1 conv(fc)로 땡겨오는 residual connection
        xf = self.dyndc(xf)
        xf = self.dropout2(xf)
        xf = self.op2(self.conv2(xf))
        return xf + self.a2 * self.rconv2(x)

#Just conv block.

class IFBlendDown(nn.Module):
    def __init__(self, in_size, rgb_in_size, out_size, dwt_size=1, dropout=0.0, default=False, device="cuda", blend=False):
        super().__init__()
        self.in_ch = in_size
        self.out_ch = out_size
        self.dwt_size = dwt_size
        self.rgb_in_size = rgb_in_size

        if dwt_size > 0:
            self.dwt = DWT_block(in_channels=in_size, out_channels=dwt_size)
        #지금 size 가지고 DWT 적용(주파수(low(평균) high(dx dy d대각)이거 각각은 1x1 conv로 연결))
        self.b_unet = UNetCompress(in_size, out_size, dropout=dropout/2)

        if default:
            self.rgb_block = BlockRGB(3, out_size, dropout_prob=dropout, device=device)
        else:
            self.rgb_block = BlockRGB(rgb_in_size, out_size, dropout_prob=dropout, device=device)

        self.fp = FusedPooling(nc=out_size, blend=blend)
    
        ##### appended
        if dwt_size > 0:
            self.mix_block = nn.Conv2d(out_size + out_size + dwt_size, out_size + out_size + dwt_size, kernel_size=1)
            #c = out_size, c_lfw = dwt_size
        else:
            self.mix_block = nn.Conv2d(out_size + out_size, out_size + out_size, kernel_size=1)
        ##### appended
        
    def forward(self, x, rgb_img):
        xu = self.b_unet(x)
        b, c, h, w = xu.shape
        rgb_feats = self.fp(self.rgb_block(rgb_img))#pooling인데 조건에 따라 conv가 뒤에 따라옴(blend일 시)

        if self.dwt_size > 0:
            lfw, hfw = self.dwt(x)
            merge_input = torch.cat((xu, rgb_feats[:, :c, :, :], lfw), dim=1)
            mixed = self.mix_block(merge_input)
            return mixed, hfw, xu, rgb_feats[:, c:, :, :]
        else:
            merge_input = torch.cat((xu, rgb_feats[:, :c, :, :]), dim=1) 
            mixed = self.mix_block(merge_input)
            return mixed, None, xu, rgb_feats[:, c:, :, :]
#이 block의 핵심은 rgb value랑 freq value를 concat 한것. , hfw->하이프리큐, xu->unet순수 결과
#rgb_img랑 x의 차이는 무엇인가. -> 깊은 레이어에서도 원본 정보를 참조하고자 쓰는게 아닐까란 추측. --> 그냥 이전에 가공한 rgb 정보를 이어서 가꼬오는데 이게 concat한 앞쪽에도 겹치잖아
#전후 둘다 레이어 놂해볼까나 ㅇㅅㅇ; 
class WASAM(nn.Module):
    '''
    Based on NAFNET Stereo Cross Attention Module (SCAM)
    '''

    def __init__(self, c_rgb, cr):
        super().__init__()
        self.scale = (0.5 * (c_rgb + cr)) ** -0.5#attention 크기 제한용용

        self.norm_l = LayerNorm2d(c_rgb)#여기도 레이어 놈을 앞뒤로해서 학습에 도움을 준다면....->원래 scam은 high freq에도 layer norm이 달려 있음. 근데 attention 방식이라 좀 애매한듯듯
        #Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3)  # B, H, c, W (transposed) 이런 방식으로 transposed로 씀.
        self.l_proj_res = nn.Conv2d(c_rgb, c_rgb // 2, kernel_size=1, stride=1, padding=0)
        self.r_proj_res = nn.Conv2d(cr, c_rgb // 2, kernel_size=1, stride=1, padding=0)
        #l->rgb, r->high for residual.. 축약해서 보냄 굳이 스킵으로 안한 이유는 모르겠네 일단 표현력이 부족한게 핵심이니까 ㅇㅅㅇ;; -> high freq랑 rgb를 'concat'해서 그럼럼
        self.l_proj1 = nn.Conv2d(c_rgb, c_rgb, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(cr, c_rgb, kernel_size=1, stride=1, padding=0)
        #attention에서 qurey 생성용 -> 쿼리가 동시에 키값에 해당됨됨 -> 그럼 콘브 하나 또 만들어서 키값 따로 뽑아뵤=죠?
        self.beta = nn.Parameter(torch.zeros((1, c_rgb // 2, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c_rgb // 2, 1, 1)), requires_grad=True)
        #learnable scale parameter
        self.l_proj2 = nn.Conv2d(c_rgb, c_rgb // 2, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(cr, c_rgb // 2, kernel_size=1, stride=1, padding=0)
        #얜 차원이 '작아지니까' value에 해당할꺼임. -> 표현력을 위해서면 굳이 작게 만들필요가 있3?
    def forward(self, x_rgb, x_hfw):
        Q_l = self.l_proj1(self.norm_l(x_rgb)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(x_hfw).permute(0, 2, 1, 3)  # B, H, c, W (transposed)

        V_l = self.l_proj2(x_rgb).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_hfw).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale
        #rgb와 hfreq의 attention을 곱함
        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l)  # B, H, W, c
        #그 attention과 value를 곱해서 유사도행렬로 만듦.
        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return torch.cat((self.l_proj_res(x_rgb) + F_r2l, self.r_proj_res(x_hfw) + F_l2r), dim=1)
        #1x1 conv를 하나 붙여서(concat이후) 진짜 'blend'를 구현하자. to do -> 이거를 위해서 

class IFBlendUp(nn.Module):
    def __init__(self, in_size, rgb_size, dwt_size,  out_size, dropout):
        super().__init__()
        self.in_ch = in_size
        self.out_ch = out_size
        self.dwt_size = dwt_size
        self.rgb_size = rgb_size

        self.b_unet = UNetDecompress(in_size + dwt_size, out_size, dropout=dropout)
        self.rgb_proj = nn.ConvTranspose2d(in_channels=rgb_size, out_channels=out_size, kernel_size=4, stride=2, padding=1)
        if dwt_size > 0:
           self.spfam = WASAM(rgb_size, dwt_size)

    def forward(self, x, hfw, rgb):
        if self.dwt_size > 0:
            rgb = self.spfam(rgb, hfw)

        state = self.b_unet(torch.cat((x, hfw), dim=1))
        state = state + F.relu(self.rgb_proj(rgb))#x_att * x_conv + x_key(q*v + k)
        return state
#decoder부분 


class IFBlend(nn.Module):
    def __init__(self, in_channels, device="cuda", use_gcb=False, blend=False):
        super().__init__()

        self.in_channels = in_channels
        self.use_gcb = use_gcb#image mode gcb

        self.in_conv = nn.Conv2d(3, in_channels, kernel_size=3, padding=1)
        self.in_bn = nn.BatchNorm2d(in_channels)

        if self.use_gcb:
            self.out_conv = nn.Sequential(
                nn.Conv2d(in_channels + 28, 3, kernel_size=7, padding=3, padding_mode="reflect")
            )
        else:
            self.out_conv = nn.Sequential(
                nn.Conv2d(in_channels, 3, kernel_size=7, padding=3, padding_mode="reflect")
            )

        if self.use_gcb:
            self.gcb = knowledge_adaptation_convnext()

        self.c1 = IFBlendDown(in_size=in_channels, rgb_in_size=3,
                              out_size=32, dwt_size=1, dropout=0.15, default=True, device=device, blend=blend)
        self.c2 = IFBlendDown(in_size=65, rgb_in_size=32, out_size=64, dwt_size=2, dropout=0.2, device=device, blend=blend)
        self.c3 = IFBlendDown(in_size=130, rgb_in_size=64, out_size=128, dwt_size=4, dropout=0.25, device=device, blend=blend)
        self.c4 = IFBlendDown(in_size=260, rgb_in_size=128, out_size=256, dwt_size=8, dropout=0.3, device=device, blend=blend)
        self.c5 = IFBlendDown(in_size=520, rgb_in_size=256, out_size=256, dwt_size=16, dropout=0.0, device=device, blend=blend)

        self.d5 = IFBlendUp(in_size=528, dwt_size=16, rgb_size=256, out_size=256, dropout=0.0)
        self.d4 = IFBlendUp(in_size=512, dwt_size=8, rgb_size=256, out_size=128, dropout=0.3)
        self.d3 = IFBlendUp(in_size=256, dwt_size=4, rgb_size=128, out_size=64, dropout=0.25)
        self.d2 = IFBlendUp(in_size=128, dwt_size=2, rgb_size=64, out_size=32, dropout=0.2)
        self.d1 = IFBlendUp(in_size=64, dwt_size=1, rgb_size=32, out_size=in_channels, dropout=0.1)

    def forward(self, x):
        x_rgb = x
        xf = self.in_bn(self.in_conv(x))
        x1, s1, xs1, rgb1 = self.c1(xf, x_rgb)
        x2, s2, xs2,  rgb2 = self.c2(x1, rgb1)
        x3, s3, xs3, rgb3 = self.c3(x2, rgb2)
        x4, s4, xs4, rgb4 = self.c4(x3, rgb3)
        x5, s5, xs5, rgb5 = self.c5(x4, rgb4)
        y5 = self.d5(x5, s5, rgb5)
        y4 = self.d4(torch.cat((y5, xs4), dim=1), s4, rgb4)
        y3 = self.d3(torch.cat((y4, xs3), dim=1), s3, rgb3)
        y2 = self.d2(torch.cat((y3, xs2), dim=1), s2, rgb2)
        y1 = self.d1(torch.cat((y2, xs1), dim=1), s1, rgb1)
        #그냥 concatenate방식으로 unet 구현함. -> 이거 표현력 높이기 위해 conv쓰고(교수님 방식) 대신 학습을 위해 앞뒤에 그거 norm 붙이자 ㅇㅇ;
        if self.use_gcb:
            return torch.sigmoid(x + self.out_conv(torch.cat((y1, self.gcb(x_rgb)), dim=1)))
        else:
            return torch.sigmoid(x + self.out_conv(y1))


if __name__ == '__main__':
    inp = torch.rand((1, 3, 512, 512))
    model = IFBlend(16, device="cpu")
    out = model(inp)
    print(out.shape)
