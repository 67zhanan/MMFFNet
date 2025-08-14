import torch
from dataset.dataset import MyRGBT_CC, MyDroneRGBT
from torch.utils.data import DataLoader
from model.MMFFNet import MMFFNet
import cv2


def gameing(output, key_points):
    game = [0, 0, 0]
    output = output.squeeze()
    key_points = key_points.squeeze()
    output = output.cpu().detach().numpy()
    H, W = key_points.shape
    ratio = H / output.shape[0]
    output = cv2.resize(output, (W, H), interpolation=cv2.INTER_CUBIC) / (ratio * ratio)
    assert output.shape == key_points.shape

    for idx, p in enumerate((2, 4, 8)):
        for i in range(p):
            for j in range(p):
                output_block = output[i * H // p:(i + 1) * H // p, j * W // p:(j + 1) * W // p]
                target_block = key_points[i * H // p:(i + 1) * H // p, j * W // p:(j + 1) * W // p]

                game[idx] += abs(output_block.sum() - target_block.sum().float())

    return game[0], game[1], game[2]


device = torch.device('cuda')
checkpoints = r'./trained_model/RGBTCC.pth'
test_path = r'./dataset/RGBT_CC/test'
test_dataset = MyRGBT_CC(test_path, train=False, game=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

model = MMFFNet().to(device)
model.load_state_dict(torch.load(checkpoints, weights_only=True))
model.eval()
game = [[], [], []]
for rgb_images, t_images, sum_person, key_points, density_maps, name in test_dataloader:
    rgb_images, t_images = rgb_images.to(device), t_images.to(device)
    outputs, outputs_norm = model(rgb_images, t_images)
    game1, game2, game3 = gameing(outputs, density_maps)
    game[0].append(game1)
    game[1].append(game2)
    game[2].append(game3)
    print(game1, game2, game3)
    # break
game1 = sum(game[0]) / len(game[0])
game2 = sum(game[1]) / len(game[2])
game3 = sum(game[2]) / len(game[2])

print(game1, game2, game3)
