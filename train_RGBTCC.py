import os
import random
import numpy as np
import torch
from dataset.dataset import MyRGBT_CC, MyDroneRGBT
from dataset.dataset import train_collate
from torch.utils.data import DataLoader
from losses.ot_loss import OT_Loss
import time
from model.MMFFNet import MMFFNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


if __name__ == '__main__':
    seed = 67
    set_seed(seed)
    epochs = 300

    rate_ot = 0.1
    rate_tv = 0.01

    batch_size = 10
    best_mae = 12
    lr = 6e-5
    idx = 0

    device = torch.device('cuda')
    log_path = rf'./logs/RGBTCC_{seed}.txt'
    save_path = r'./trained_model/RGBTCC'
    checkpoints = r''
    train_path = r'./dataset/RGBT_CC/train'
    val_path = r'./dataset/RGBT_CC/val'
    f = open(log_path, 'w')

    train_dataset = MyRGBT_CC(train_path, train=True, crop_size=256)
    val_dataset = MyRGBT_CC(val_path, train=False)
    train_dataloader = DataLoader(train_dataset, collate_fn=train_collate,
                                  batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    model = MMFFNet().to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if checkpoints != r'':
        model.load_state_dict(torch.load(checkpoints, weights_only=True))

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    count_loss = torch.nn.L1Loss().to(device)
    ot_loss = OT_Loss(256, 8, 0, device, 100, 10)
    tv_loss = torch.nn.L1Loss(reduction='none').to(device)

    for epoch in range(1, epochs + 1):
        print(f'---------第{epoch}次训练开始---------')
        model.train()
        t1 = time.time()
        for rgb_images, t_images, key_points, density in train_dataloader:
            rgb_images, t_images = rgb_images.to(device), t_images.to(device)
            sum_person = np.array([len(p) for p in key_points], dtype=np.float32)  # 总人数
            key_points = [p.to(device) for p in key_points]  # 标注点
            density = density.to(device)  # 密度图
            batch_size = rgb_images.size(0)  # 批量大小

            outputs, outputs_norm = model(rgb_images, t_images)

            # ot_loss
            ot_loss_result, wd, value = ot_loss(outputs_norm, outputs, key_points)

            # tv_loss
            sum_person_tensor = (torch.from_numpy(sum_person).float().to(device).unsqueeze(1).unsqueeze(2).unsqueeze(3))
            density_norm = density / (sum_person_tensor + 1e-6)
            tv_loss_result = (tv_loss(outputs_norm, density_norm).sum(1).sum(1).sum(1)
                            * torch.from_numpy(sum_person).float().to(device)).mean(0)

            # count_loss
            count_loss_result = count_loss(outputs.sum(1).sum(1).sum(1),
                                        torch.from_numpy(sum_person).float().to(device))

            loss = (
                    count_loss_result
                    + ot_loss_result * rate_ot
                    + tv_loss_result * rate_tv
                    )

            optim.zero_grad()
            loss.backward()
            optim.step()
            # break
        t2 = time.time()
        print('第{0:d}轮次训练完成，花费{1:.2f}秒'.format(epoch, (t2 - t1)))

        t1 = time.time()
        model.eval()
        epoch_res = []
        for rgb_images, t_images, sum_person, name in val_dataloader:
            with torch.no_grad():
                rgb_images, t_images = rgb_images.to(device), t_images.to(device)

                outputs, outputs_norm = model(rgb_images, t_images)
                res = sum_person[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)

                # break

        t2 = time.time()
        epoch_res = np.array(epoch_res)
        Rmse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), os.path.join(save_path, f'{epochs}_{seed}_{idx}.pth'))
            idx += 1

        print('第{0:d}轮验证完成, 花费{1:.2f}秒, mae:{2:.2f}, Rmse:{3:.2f}, best_mae:{4:.2f}'.format(epoch, t2 - t1, mae, Rmse, best_mae))
        f.write('epoch:{0:d}, mae:{1:.2f}, Rmse:{2:.2f}, best_mae:{3:.2f}'.format(epoch, mae, Rmse, best_mae))
        f.write('\n')
        torch.cuda.empty_cache()

    f.close()
    print('训练完成')