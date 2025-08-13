from PIL import Image
from lxml import etree
import os
import glob
import numpy as np


path = ["./DroneRGBT/Train/GT_", "./DroneRGBT/Test/GT_"]
for root in path:
    filepath = glob.glob(os.path.join(root, '*.xml'))

    for filename in filepath:
        # 使用r模式打开文件，编码方式为utf-8
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        # 读取XML文档内容，返回选择器
        selector = etree.XML(content)  # 注意这里使用XML而不是HTML，因为.xml是XML格式的文件
        # 使用XPath查找所有的point元素
        points = selector.xpath('//object/point')

        point_list = []
        # 遍历所有找到的point元素，提取x和y坐标
        for point in points:
            x = int(point.find('x').text)
            y = int(point.find('y').text)
            point_list.append([x, y])
            # print(f"Point: ({x}, {y})")

        points_array = np.array(point_list)
        base_name = os.path.splitext(os.path.basename(filename))[0]

        # 保存NumPy数组为.npy文件
        npy_filename = os.path.join(root, base_name + '.npy')
        np.save(npy_filename, points_array)
        print(f"Saved points to {npy_filename}")

