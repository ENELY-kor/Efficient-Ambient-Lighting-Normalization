import os
import numpy as np
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from random import random, choice
from PIL import Image
from torchvision.transforms import InterpolationMode



class ImageSet(data.Dataset):
    def __init__(self, set_path, set_type, aug=True, size=(512, 512), mode='rcrop'):
        path_dir = '{}/{}'.format(set_path, set_type)

        self.mode = mode
        self.size = size
        self.aug = aug

        self.gt_list = []
        self.inp_list = []
        self.num_samples = 0

        file_list = [f for f in os.listdir(path_dir) if f.endswith("gt.png")]

        for f in file_list:
            inp_path = os.path.join(path_dir, f.replace("gt", "in"))
            gt_path = os.path.join(path_dir, f)

            self.inp_list.append(inp_path)
            self.gt_list.append(gt_path)
            self.num_samples += 1

        assert len(self.inp_list) == len(self.gt_list)
        assert len(self.inp_list) == self.num_samples

    def __len__(self):
        return self.num_samples

    def load_image_tensor(img_path, size=(512, 512)):
        img = Image.open(img_path).convert('RGB')
        img = torchvision.transforms.functional.resize(img, size)
        return torchvision.transforms.functional.to_tensor(img)

    def rand_bbox(W, H, lam):
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    def augs(self, inp, gt):
        if self.mode == 'rcrop':
            w, h = gt.size
            tl = np.random.randint(0, h - self.size[0])
            tt = np.random.randint(0, w - self.size[1])

            gt = torchvision.transforms.functional.crop(gt, tt, tl, self.size[0], self.size[1])
            inp = torchvision.transforms.functional.crop(inp, tt, tl, self.size[0], self.size[1])
        else:
            to_pil = transforms.ToPILImage()
            if random() < 0.5:#####appended 안섞이더라도 모든 데이터는 학습 할 수 있게.
                idx1, idx2 = random.sample(range(len(file_list)), 2)
                f1 = file_list[idx1]
                f2 = file_list[idx2]
            
                inp1_path = os.path.join(path_dir, f1.replace("gt", "in"))
                gt1_path = os.path.join(path_dir, f1)
                inp2_path = os.path.join(path_dir, f2.replace("gt", "in"))
                gt2_path = os.path.join(path_dir, f2)
                
                inp1 = inp#load_image_tensor(inp1_path)
                gt1 = gt#load_image_tensor(gt1_path)
                inp2 = load_image_tensor(inp2_path)
                gt2 = load_image_tensor(gt2_path)
                
                alpha = 0.4  #mixup hyper parameter
                lam = np.random.beta(alpha, alpha)
                inp = to_pil(lam * inp1 + (1 - lam) * inp2)
                gt  = to_pil(lam * gt1 + (1 - lam) * gt2)
            if random() < 0.5:#####appended
                idx1, idx2 = random.sample(range(len(file_list)), 2)
                f1 = file_list[idx1]
                f2 = file_list[idx2]
            
                inp1_path = os.path.join(path_dir, f1.replace("gt", "in"))
                gt1_path = os.path.join(path_dir, f1)
                inp2_path = os.path.join(path_dir, f2.replace("gt", "in"))
                gt2_path = os.path.join(path_dir, f2)
                
                inp1, gt1 = inp, gt#load_img(inp1_path), load_img(gt1_path)
                inp2, gt2 = load_img(inp2_path), load_img(gt2_path)
                C, H, W = inp1.shape
                alpha = 1.0
                lam = np.random.beta(alpha, alpha)
                
                bbx1, bby1, bbx2, bby2 = rand_bbox(W, H, lam)
                mixed_inp = inp1.clone()
                mixed_inp[:, bby1:bby2, bbx1:bbx2] = inp2[:, bby1:bby2, bbx1:bbx2]
                mixed_gt = gt1.clone()
                mixed_gt[:, bby1:bby2, bbx1:bbx2] = gt2[:, bby1:bby2, bbx1:bbx2]
                inp = to_pil(mixed_inp)
                gt = to_pil(mixed_gt)
            gt = torchvision.transforms.functional.resize(gt, self.size, InterpolationMode.BICUBIC)
            inp = torchvision.transforms.functional.resize(inp, self.size, InterpolationMode.BICUBIC)

        if random() < 0.5:
            inp = torchvision.transforms.functional.hflip(inp)
            gt = torchvision.transforms.functional.hflip(gt)
        if random() < 0.5:
            inp = torchvision.transforms.functional.vflip(inp)
            gt = torchvision.transforms.functional.vflip(gt)
        if random() < 0.5:
            angle = choice([90, 180, 270])
            inp = torchvision.transforms.functional.rotate(inp, angle)
            gt = torchvision.transforms.functional.rotate(gt, angle)
        ##########appended
        if random() < 0.5:
            tmp_inp = torchvision.transforms.functional.invert(gt)
            gt = torchvision.transforms.functional.invert(inp)#inverted color
            inp = tmp_inp #색상 반전의 이유는 빛을 가하는 형태를 학습하기 위해 -> 완전 없애는것보단 0.1정도로 있는게 좋은듯?
        #if random() < 0:
            #inp = torchvision.transforms.functional.adjust_sharpness(inp, sharpness_factor=0.5) # gaussian image
            #gt = torchvision.transforms.functional.adjust_sharpness(gt, sharpness_factor=0.5)       # sharpened gt
        if random() < 0.1:
            inp = torchvision.transforms.functional.adjust_sharpness(inp, sharpness_factor=2.0) # sharpened image
            #gt = torchvision.transforms.functional.adjust_sharpness(gt, sharpness_factor=2.0)       # sharpened gt    
    #샤프닝의 이유는 테두리에 더 가산을 둬서 지울 수 있게 ->  샤프닝이랑 뭉게는 거랑 둘 다 효과 없어서 둘 다 적용해보기 ==> gt를 왜곡하면 어캄;
    #회전 플립 말고 CUTMIX의 AUG도 넣자. torchvision.transforms.v2.MixUp
        return inp, gt

    def __getitem__(self, index):
        inp_data = Image.open(self.inp_list[index])
        gt_data = Image.open(self.gt_list[index])

        to_tensor = transforms.ToTensor()

        if self.aug:
            inp_data, gt_data = self.augs(inp_data, gt_data)
        else:
            if self.size is not None:
                inp_data = torchvision.transforms.functional.resize(inp_data, self.size, InterpolationMode.BICUBIC)
                gt_data = torchvision.transforms.functional.resize(gt_data, self.size, InterpolationMode.BICUBIC)

        return to_tensor(inp_data), to_tensor(gt_data)


class ISTDImageSet(data.Dataset):
    def __init__(self, set_path, set_type, size=(256, 256), use_mask=True, aug=False):
        self.augment = aug

        self.size = size
        self.use_mask = use_mask

        self.to_tensor = transforms.ToTensor()
        if size is not None:
            self.resize = transforms.Resize(self.size, interpolation=InterpolationMode.BICUBIC)
        else:
            self.resize = None

        clean_path_dir = '{}/{}/{}_C'.format(set_path, set_type, set_type)

        self.gt_images_path = []
        self.masks_path = []
        self.inp_images_path = []
        self.num_samples = 0

        for dirpath, dnames, fnames \
                in os.walk("{}/{}/{}_A/".format(set_path, set_type, set_type)):
            for f in fnames:
                if f.endswith(".zip"):
                    continue
                orig_path = os.path.join(dirpath, f)
                clean_path = os.path.join(clean_path_dir, f)

                self.gt_images_path.append(clean_path)
                self.inp_images_path.append(orig_path)

                self.num_samples += 1

    def __len__(self):
        return self.num_samples

    def augs(self, gt, inp):
        w, h = gt.size
        tl = np.random.randint(0, h - self.size[0])
        tt = np.random.randint(0, w - self.size[1])

        gt = torchvision.transforms.functional.crop(gt, tt, tl, self.size[0], self.size[1])
        inp = torchvision.transforms.functional.crop(inp, tt, tl, self.size[0], self.size[1])

        if random() < 0.5:
            inp = torchvision.transforms.functional.hflip(inp)
            gt = torchvision.transforms.functional.hflip(gt)
        if random() < 0.5:
            inp = torchvision.transforms.functional.vflip(inp)
            gt = torchvision.transforms.functional.vflip(gt)
        if random() < 0.5:
            angle = choice([90, 180, 270])
            inp = torchvision.transforms.functional.rotate(inp, angle)
            gt = torchvision.transforms.functional.rotate(gt, angle)

        return gt, inp

    def __getitem__(self, index):
        inp_data = Image.open(self.inp_images_path[index])
        gt_data = Image.open(self.gt_images_path[index])

        if self.augment:
            gt_data, inp_data = self.augs(gt_data, inp_data)
        else:
            if self.resize is not None:
                gt_data = self.resize(gt_data)
                inp_data = self.resize(inp_data)

        tensor_gt = self.to_tensor(gt_data)
        tensor_inp = self.to_tensor(inp_data)

        return tensor_inp, tensor_gt

#같은 내용을 istd image set에 한정(그림자 지우기를 위한 한정된 데이터 셋)

class ISTDImageMaskSet(data.Dataset):
    def __init__(self, set_path, set_type, size=(256, 256), use_mask=True, aug=False):
        self.augment = aug

        self.size = size
        self.use_mask = use_mask

        self.to_tensor = transforms.ToTensor()
        if size is not None:
            self.resize = transforms.Resize(self.size, interpolation=InterpolationMode.BICUBIC)
        else:
            self.resize = None

        clean_path_dir = '{}/{}/{}_C'.format(set_path, set_type, set_type)
        mask_path_dir = '{}/{}/{}_B'.format(set_path, set_type, set_type)

        self.gt_images_path = []
        self.masks_path = []
        self.inp_images_path = []
        self.num_samples = 0

        for dirpath, dnames, fnames \
                in os.walk("{}/{}/{}_A/".format(set_path, set_type, set_type)):
            for f in fnames:
                if f.endswith(".zip"):
                    continue
                orig_path = os.path.join(dirpath, f)
                clean_path = os.path.join(clean_path_dir, f)
                mask_path = os.path.join(mask_path_dir, f)

                self.gt_images_path.append(clean_path)
                self.inp_images_path.append(orig_path)
                self.masks_path.append(mask_path)

                self.num_samples += 1

    def __len__(self):
        return self.num_samples

    def augs(self, gt, inp):
        w, h = gt.size
        tl = np.random.randint(0, h - self.size[0])
        tt = np.random.randint(0, w - self.size[1])

        gt = torchvision.transforms.functional.crop(gt, tt, tl, self.size[0], self.size[1])
        inp = torchvision.transforms.functional.crop(inp, tt, tl, self.size[0], self.size[1])

        if random() < 0.5:
            inp = torchvision.transforms.functional.hflip(inp)
            gt = torchvision.transforms.functional.hflip(gt)
        if random() < 0.5:
            inp = torchvision.transforms.functional.vflip(inp)
            gt = torchvision.transforms.functional.vflip(gt)
        if random() < 0.5:
            angle = choice([90, 180, 270])
            inp = torchvision.transforms.functional.rotate(inp, angle)
            gt = torchvision.transforms.functional.rotate(gt, angle)

        return gt, inp

    def __getitem__(self, index):
        inp_data = Image.open(self.inp_images_path[index])
        gt_data = Image.open(self.gt_images_path[index])
        smat_data = Image.open(self.masks_path[index])

        if self.resize is not None:
            gt_data = self.resize(gt_data)
            inp_data = self.resize(inp_data)
            smat_data = self.resize(smat_data)


        tensor_gt = self.to_tensor(gt_data)
        tensor_mask = self.to_tensor(smat_data)
        tensor_inp = self.to_tensor(inp_data)

        return tensor_inp, tensor_mask, tensor_gt
#같은 내용을 istd "masked" image set에 한정(그림자 지우기를 위한 한정된 데이터 셋)
