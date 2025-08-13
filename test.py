import os

import torch
import numpy as np
from dataset.dataset import MyRGBT_CC, MyDroneRGBT
from torch.utils.data import DataLoader
from model.MMFFNet import MMFFNet


save_dir = r'./logs/test.txt'
if save_dir != r'':
    f = open(save_dir, 'w')
device = torch.device('cuda')
weight_path = "./trained_model/RGBTCC"
test_path = r'./dataset/RGBT_CC/test'
test_dataset = MyRGBT_CC(test_path, train=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

model = MMFFNet().to(device)
model.eval()
for weight in os.listdir(weight_path):
    epoch_res = []
    i = 0
    model.load_state_dict(torch.load(os.path.join(weight_path, weight), weights_only=True))
    for rgb_images, t_images, sum_person, name, in test_dataloader:
        with torch.no_grad():
            rgb_images, t_images = rgb_images.to(device), t_images.to(device)
            outputs, outputs_norm = model(rgb_images, t_images)
            res = sum_person[0].item() - torch.sum(outputs).item()
            epoch_res.append(res)
            i += 1
            print(f'{i}, 误差:{res:.2f}, 图片名:{name[0]}, 实际总人数:{sum_person[0].item()}, 预测总人数:{torch.sum(outputs).item():.2f}')
            if save_dir != r'':
                f.write(f'mae: {res:.2f}, name: {name[0]}, truth: {sum_person[0].item()}, perdict: {torch.sum(outputs).item():.2f} \n')
            break

    epoch_res = np.array(epoch_res)
    Rmse = np.sqrt(np.mean(np.square(epoch_res)))
    mae = np.mean(np.abs(epoch_res))
    print('mae:{0:.2f}, Rmse:{1:.2f}'.format(mae, Rmse))
if save_dir != r'':
    f.close()