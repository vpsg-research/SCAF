import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import albumentations as albu
from albumentations.core.transforms_interface import DualTransform
import cv2


class RandomCopyMove(DualTransform):
    def __init__(self,
        max_h = 0.8,
        max_w = 0.8,
        min_h = 0.05,
        min_w = 0.05,
        mask_value = 255,
        always_apply = False,
        p = 0.5,  
    ):
        """Apply cope-move manipulation to the image, and change the respective region on the mask to <mask_value>

        Args:
            max_h (float, optional): (0~1), max window height rate to the full height of image . Defaults to 0.5.
            max_w (float, optional): (0~1), max window width rate to the full width of image . Defaults to 0.5.
            min_h (float, optional): (0~1), min window height rate to the full height of image . Defaults to 0.05.
            min_w (float, optional): (0~1), min window width rate to the full width of image . Defaults to 0.05.
            mask_value (int, optional): the value apply the tampered region on the mask. Defaults to 255.
            always_apply (bool, optional): _description_. Defaults to False.
            p (float, optional): _description_. Defaults to 0.5.
        """
        super(RandomCopyMove, self).__init__(always_apply, p)
        self.max_h = max_h
        self.max_w = max_w
        self.min_h = min_h
        self.min_w = min_w
        self.mask_value = mask_value
        
    def _get_random_window(
        self, 
        img_height, 
        img_width, 
        window_height = None, 
        window_width = None
    ):
        assert self.max_h < 1 and self.max_h > 0 
        assert self.max_w < 1 and self.max_w > 0
        assert self.min_w < 1 and self.min_w > 0
        assert self.min_h < 1 and self.min_h > 0
        
        l_min_h = int(img_height * self.min_h)
        l_min_w = int(img_width * self.min_w)
        l_max_h = int(img_height * self.max_h)
        l_max_w = int(img_width * self.max_w)
        
        if window_width == None or window_height == None:
            window_h = np.random.randint(l_min_h, l_max_h)
            window_w = np.random.randint(l_min_w, l_max_w)
        else:
            window_h = window_height
            window_w = window_width

        # position of left up corner of the window
        pos_h = np.random.randint(0, img_height - window_h)
        pos_w = np.random.randint(0, img_width - window_w)
        
        return pos_h, pos_w , window_h, window_w
        
        
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        image = img.copy()
        H, W, _ = image.shape
        # copy region:
        c_pos_h, c_pos_w, c_window_h, c_window_w = self._get_random_window(H, W)
        
        # past region, window size is defined by copy region:
        self.p_pos_h, self.p_pos_w, self.p_window_h, self.p_window_w = self._get_random_window(H, W, c_window_h, c_window_w)
          
        copy_region = image[
            c_pos_h: c_pos_h + c_window_h, 
            c_pos_w: c_pos_w + c_window_w, 
            : 
        ]
        image[
            self.p_pos_h : self.p_pos_h + self.p_window_h,
            self.p_pos_w : self.p_pos_w + self.p_window_w,
            :
        ] = copy_region
        return image
        

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        """
        change the mask of manipulated region to 1
        """
    
        manipulated_region =  np.full((self.p_window_h, self.p_window_w), 1)
        img = img.copy()
        img[
            self.p_pos_h : self.p_pos_h + self.p_window_h,
            self.p_pos_w : self.p_pos_w + self.p_window_w,
        ] = self.mask_value
        return img
        
class RandomInpainting(DualTransform):
    def __init__(self,
        max_h = 0.8,
        max_w = 0.8,
        min_h = 0.05,
        min_w = 0.05,
        mask_value = 255,
        always_apply = False,
        p = 0.5,  
    ):
        super(RandomInpainting, self).__init__(always_apply, p)
        self.max_h = max_h
        self.max_w = max_w
        self.min_h = min_h
        self.min_w = min_w
        self.mask_value = mask_value
    def _get_random_window(
        self, 
        img_height, 
        img_width, 
    ):
        assert self.max_h < 1 and self.max_h > 0 
        assert self.max_w < 1 and self.max_w > 0
        assert self.min_w < 1 and self.min_w > 0
        assert self.min_h < 1 and self.min_h > 0
        
        l_min_h = int(img_height * self.min_h)
        l_min_w = int(img_width * self.min_w)
        l_max_h = int(img_height * self.max_h)
        l_max_w = int(img_width * self.max_w)
    
        window_h = np.random.randint(l_min_h, l_max_h)
        window_w = np.random.randint(l_min_w, l_max_w)

        # position of left up corner of the window
        pos_h = np.random.randint(0, img_height - window_h)
        pos_w = np.random.randint(0, img_width - window_w)
        
        return pos_h, pos_w , window_h, window_w
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img = img.copy()
        img = np.uint8(img)
        H, W, C = img.shape
        mask = np.zeros((H, W), dtype=np.uint8)
        # inpainting region
        self.pos_h, self.pos_w , self.window_h, self.window_w = self._get_random_window(H, W)
        mask[
            self.pos_h : self.pos_h+ self.window_h,
            self.pos_w : self.pos_w + self.window_w,
        ] = 1
        inpaint_flag = cv2.INPAINT_TELEA if random.random() > 0.5 else cv2.INPAINT_NS
        img = cv2.inpaint(img, mask, 3,inpaint_flag)
        return img
    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        """
        change the mask of manipulated region to 1
        """
        img = img.copy()
        img[
            self.pos_h : self.pos_h+ self.window_h,
            self.pos_w : self.pos_w + self.window_w,
        ] = self.mask_value
        return img

# 自定义的 PolypObjDataset 类
class PolypObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, edge_root, trainsize, type_='train'):
        self.trainsize = trainsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.tif') or f.endswith('.bmp')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')]
        self.egs = [os.path.join(edge_root, f) for f in os.listdir(edge_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.egs = sorted(self.egs)

        self.filter_files()

        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])
        self.eg_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])

        # albumentations transforms
        if type_ == 'train':
            self.trans = albu.Compose([
                albu.RandomScale(scale_limit=0.2, p=1),
                RandomCopyMove(p=0.1),
                RandomInpainting(p=0.1),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=0.1, p=1),
                albu.ImageCompression(quality_lower=70, quality_upper=100, p=0.2),
                albu.RandomRotate90(p=0.5),
                albu.GaussianBlur(blur_limit=(3, 7), p=0.2)
            ])
        else:
            self.trans = albu.Compose([])

        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        eg = self.binary_loader(self.egs[index])

        W, H = gt.size    # PIL: (width, height)

        augmented = self.trans(image=np.array(image), masks=[np.array(gt), np.array(eg)])
        image, masks = augmented['image'], augmented['masks']
        gt, eg = masks[0], masks[1]

        image = Image.fromarray(image)  # 将numpy数组转换为PIL图像
        gt = Image.fromarray(gt)        # 将numpy数组转换为PIL图像
        eg = Image.fromarray(eg)        # 将numpy数组转换为PIL图像

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        gt[gt == 0.] = 255.
        gt[gt == 2.] = 0.
        eg = self.eg_transform(eg)

        return image, gt, (H, W), 'None'
        # imagepath, maskpath = self.samples[idx]
        # image               = cv2.imread(imagepath).astype(np.float32)[:,:,::-1]
        # mask                = cv2.imread(maskpath).astype(np.float32)[:,:,::-1]
        # H, W, C             = mask.shape
        # if self.cfg.mode == 'train':
        #     image, mask         = self.transform(image, mask)
        #     mask[mask == 0.] = 255.
        #     mask[mask == 2.] = 0.
        # else:
        #     image, _         = self.transform(image, mask)
        #     mask = torch.from_numpy(mask.copy()).permute(2,0,1)
        #     mask = mask.mean(dim=0, keepdim=True)
        #     mask /= 255
        # # print(image.max(), image.min())
        # return image, mask, (H, W), maskpath.split('/')[-1]

    def filter_files(self):
        print(len(self.images))
        print(len(self.gts))
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.egs)
        images = []
        gts = []
        egs = []
        for img_path, gt_path, eg_path in zip(self.images, self.gts, self.egs):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            eg = Image.open(eg_path)
            if img.size == gt.size == eg.size:
                images.append(img_path)
                gts.append(gt_path)
                egs.append(eg_path)
        print(len(images))
        print(len(gts))
        self.images = images
        self.gts = gts
        self.egs = egs

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, gt_root, edge_root, batchsize, trainsize, type_='train',
               shuffle=True, num_workers=12, pin_memory=True):
    dataset = PolypObjDataset(image_root, gt_root, edge_root, trainsize, type_)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader



# test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif') or f.endswith('.bmp')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png') or f.endswith('.tif') or f.endswith('.bmp')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        print(self.size)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])

        name = self.images[self.index].split('/')[-1]

        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        return image, gt, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
