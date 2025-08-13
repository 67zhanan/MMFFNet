import torch
from PIL import ImageDraw
from torch.utils.data import DataLoader, Dataset
import glob
import os
import torchvision.transforms as trans
import numpy as np
import random
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.transforms import ToPILImage


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def gen_discrete_map(im_height, im_width, points):
    """
        func: generate the discrete map.
        points: [num_gt, 2], for each row: [width, height]
        """
    discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
    h, w = discrete_map.shape[:2]
    num_gt = points.shape[0]
    if num_gt == 0:
        return discrete_map

    # fast create discrete map
    points_np = np.array(points).round().astype(int)
    p_h = np.minimum(points_np[:, 1], np.array([h - 1] * num_gt).astype(int))
    p_w = np.minimum(points_np[:, 0], np.array([w - 1] * num_gt).astype(int))
    p_index = torch.from_numpy(p_h * im_width + p_w).to(torch.int64)
    discrete_map = torch.zeros(im_width * im_height).scatter_add_(0, index=p_index,
                                                                  src=torch.ones(im_width * im_height)).view(im_height,
                                                                                                             im_width).numpy()

    ''' slow method
    for p in points:
        p = np.round(p).astype(int)
        p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
        discrete_map[p[0], p[1]] += 1
    '''
    assert np.sum(discrete_map) == num_gt
    return discrete_map


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    rgb_images = torch.stack(transposed_batch[0], 0)
    t_images = torch.stack(transposed_batch[1], 0)
    points = transposed_batch[2]  # the number of points is not fixed, keep it as a list of tensor
    gt_discretes = torch.stack(transposed_batch[3], 0)
    return rgb_images, t_images, points, gt_discretes


class MyRGBT_CC(Dataset):
    def __init__(self, root, train, crop_size=256, game=False):
        self.root = root
        self.RGB_path = glob.glob(os.path.join(self.root, '*RGB.jpg'))
        self.T_path = glob.glob(os.path.join(self.root, '*T.jpg'))
        self.GT_path = glob.glob(os.path.join(self.root, '*.npy'))
        self.train = train
        self.game = game
        self.crop_size = crop_size
        self.RGB_transform = trans.Compose([trans.ToTensor(),
                                            trans.Normalize(mean=[0.407, 0.389, 0.396],
                                                            std=[0.241, 0.246, 0.242]),])
        self.T_transform = trans.Compose([trans.ToTensor(),
                                          trans.Normalize(mean=[0.492, 0.168, 0.430],
                                                          std=[0.317, 0.174, 0.191]),])

    def __len__(self):
        return len(self.RGB_path)

    def __getitem__(self, item):
        RGB_image = Image.open(self.RGB_path[item]).convert('RGB')
        T_image = Image.open(self.T_path[item]).convert('RGB')
        key_points = np.load(self.GT_path[item])
        if self.train:
            w, h = RGB_image.size

            # 随机裁剪
            start_y, start_x, h_size, w_size = random_crop(h, w, self.crop_size, self.crop_size)
            RGB_image = F.crop(RGB_image, start_y, start_x, h_size, w_size)
            T_image = F.crop(T_image, start_y, start_x, h_size, w_size)
            if len(key_points) > 0:
                key_points -= [start_x, start_y]
                id_mask = ((key_points[:, 0] > 0) * (key_points[:, 0] < w_size) *
                           (key_points[:, 1] > 0) * (key_points[:, 1] < h_size))
                key_points = key_points[id_mask]
            else:
                key_points = np.empty([0, 2])

            # 生成掩码图
            density = gen_discrete_map(h_size, w_size, key_points)
            h_down = h_size // 8
            w_down = w_size // 8
            density = density.reshape([h_down, 8, w_down, 8]).sum(axis=(1, 3))

            # 随机水平翻转
            if random.random() > 0.5:
                RGB_image = F.hflip(RGB_image)
                T_image = F.hflip(T_image)
                density = np.fliplr(density)
                if len(key_points) > 0:
                    key_points[:, 0] = w_size - 1 - key_points[:, 0]
            density = np.expand_dims(density, 0)

            return self.RGB_transform(RGB_image), self.T_transform(T_image), torch.from_numpy(key_points.copy()).float(), torch.from_numpy(density.copy()).float()
        else:
            file = self.RGB_path[item]
            name = os.path.splitext(os.path.basename(file))[0]
            if self.game:
                w, h = RGB_image.size
                density = gen_discrete_map(h, w, key_points)
                return self.RGB_transform(RGB_image), self.T_transform(T_image), len(key_points), torch.from_numpy(key_points.copy()).float(), density, name
            else:
                return self.RGB_transform(RGB_image), self.T_transform(T_image), len(key_points), name


class MyDroneRGBT(Dataset):
    def __init__(self, root, train, crop_size=256, game=False, val=False):
        self.root = root
        self.RGB_path = glob.glob(os.path.join(self.root, 'RGB', '*.jpg'))
        self.T_path = glob.glob(os.path.join(self.root, 'Infrared', '*.jpg'))
        self.GT_path = glob.glob(os.path.join(self.root, 'GT_', '*.npy'))
        self.train = train
        self.val = val
        self.game = game
        self.crop_size = crop_size
        self.RGB_transform = trans.Compose([trans.ToTensor(),
                                            trans.Normalize(mean=[0.3111, 0.3107, 0.3281],
                                                            std=[0.233, 0.2227, 0.2332]),])
        # self.RGB_transform = trans.Compose([trans.ToTensor(),
        #                                     trans.Normalize(mean=[0.407, 0.389, 0.396],
        #                                                     std=[0.241, 0.246, 0.242]),])
        self.T_transform = trans.Compose([trans.ToTensor(),
                                          trans.Normalize(mean=[0.474, 0.474, 0.474],
                                                          std=[0.1716, 0.1716, 0.1716]),])
        # self.T_transform = trans.Compose([trans.ToTensor(),
        #                                   trans.Normalize(mean=[0.492, 0.168, 0.430],
        #                                                   std=[0.317, 0.174, 0.191]),])

    def __len__(self):
        return len(self.RGB_path)

    def __getitem__(self, item):
        RGB_image = Image.open(self.RGB_path[item]).convert('RGB')
        T_image = Image.open(self.T_path[item]).convert('RGB')
        key_points = np.load(self.GT_path[item])
        if self.train:
            w, h = RGB_image.size

            # 随机裁剪
            start_y, start_x, h_size, w_size = random_crop(h, w, self.crop_size, self.crop_size)
            RGB_image = F.crop(RGB_image, start_y, start_x, h_size, w_size)
            T_image = F.crop(T_image, start_y, start_x, h_size, w_size)
            if len(key_points) > 0:
                key_points -= [start_x, start_y]
                id_mask = ((key_points[:, 0] > 0) * (key_points[:, 0] < w_size) *
                           (key_points[:, 1] > 0) * (key_points[:, 1] < h_size))
                key_points = key_points[id_mask]
            else:
                key_points = np.empty([0, 2])

            # 生成掩码图
            density = gen_discrete_map(h_size, w_size, key_points)
            h_down = h_size // 8
            w_down = w_size // 8
            density = density.reshape([h_down, 8, w_down, 8]).sum(axis=(1, 3))

            # 随机水平翻转
            if random.random() > 0.5:
                RGB_image = F.hflip(RGB_image)
                T_image = F.hflip(T_image)
                density = np.fliplr(density)
                if len(key_points) > 0:
                    key_points[:, 0] = w_size - 1 - key_points[:, 0]
            density = np.expand_dims(density, 0)

            return self.RGB_transform(RGB_image), self.T_transform(T_image), torch.from_numpy(key_points.copy()).float(), torch.from_numpy(density.copy()).float()
        else:
            file = self.RGB_path[item]
            name = os.path.splitext(os.path.basename(file))[0]
            if self.game:
                w, h = RGB_image.size
                density = gen_discrete_map(h, w, key_points)
                return self.RGB_transform(RGB_image), self.T_transform(T_image), len(key_points), torch.from_numpy(key_points.copy()).float(), density, name
            else:
                return self.RGB_transform(RGB_image), self.T_transform(T_image), len(key_points), name


if __name__ == '__main__':
    rgbtcc_dateset = MyRGBT_CC(r'D:\ZhouKunyu\dataset\RGBT_CC\val', train=True, crop_size=256)
    drone_dataset = MyDroneRGBT(r'D:\ZhouKunyu\dataset\DroneRGBT\Train', train=True, crop_size=256)
    dataloader = DataLoader(drone_dataset,
                            # collate_fn=train_collate,
                            batch_size=1, shuffle=False, num_workers=4, drop_last=True)
    to_pil_image = ToPILImage()
    for rgb_images, t_images, key_points, density in dataloader:
        rgb_images = rgb_images.squeeze()
        t_images = t_images.squeeze()
        point = key_points.squeeze()
        # 检查Tensor值的范围
        min_val = rgb_images.min()
        max_val = rgb_images.max()

        min_t = t_images.min()
        max_t = t_images.max()
        # 归一化Tensor到[0, 1]范围
        # 这里我们假设Tensor的值范围是[min_val, max_val]
        normalized_tensor = (rgb_images - min_val) / (max_val - min_val)
        t_image = (t_images - min_t) / (max_t - min_t)

        pil_image = to_pil_image(normalized_tensor)
        t_image = to_pil_image(t_image)

        draw1 = ImageDraw.Draw(pil_image)
        draw2 = ImageDraw.Draw(t_image)
        # 遍历点数组，并在图像上绘制每个点
        for points in point:
        # 绘制较大的点，使用ellipse方法绘制一个小圆作为点的标记
        # 这里设置了一个半径为5的圆，颜色为红色
            x, y = points
            draw1.ellipse((x - 5, y - 5, x + 5, y + 5), fill='red')
            draw2.ellipse((x - 5, y - 5, x + 5, y + 5), fill='red')
        pil_image.show()
        t_image.show()
        print(rgb_images.shape, t_images.shape, len(key_points), density.shape)
        break
    # test_dataset = MyShanghai(r'D:\ZhouKunyu\MyNet\shanghaitech\train_data\train_img',
    #                           r'D:\ZhouKunyu\MyNet\shanghaitech\train_data\train_depth_npy',
    #                           r'D:\ZhouKunyu\MyNet\shanghaitech\train_data\train_bbox_npy',
    #                           train=True,
    #                           crop_size=768)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    # to_pil_image = ToPILImage()
    # for i, (rgb_images, t_images, point, den) in enumerate(test_dataloader):
    #
    #     rgb_images = rgb_images.squeeze()
    #     point = point.squeeze()
    #     # 检查Tensor值的范围
    #     min_val = rgb_images.min()
    #     max_val = rgb_images.max()
    #
    #     # 归一化Tensor到[0, 1]范围
    #     # 这里我们假设Tensor的值范围是[min_val, max_val]
    #     normalized_tensor = (rgb_images - min_val) / (max_val - min_val)
    #
    #     pil_image = to_pil_image(normalized_tensor)
    #     draw = ImageDraw.Draw(pil_image)
    #     # 遍历点数组，并在图像上绘制每个点
    #     for points in point:
    #         # 绘制较大的点，使用ellipse方法绘制一个小圆作为点的标记
    #         # 这里设置了一个半径为5的圆，颜色为红色
    #         x, y = points
    #         draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill='red')
    #     pil_image.show()
    #     break

        # print(rgb_images[0].shape, t_images[0].shape, point[0].shape, den[0].shape)
        # if i == 5:
        #     break
