# from random import shuffle
# import numpy as np
# import torch
# import torch.nn as nn
# import math
# import torch.nn.functional as F
# from PIL import Image
# from torch.autograd import Variable
# from torch.utils1.data import DataLoader
# from torch.utils1.data.dataset import Dataset
# from utils1.utils1 import bbox_iou, merge_bboxes
# # from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
# #from nets.yolo_training import Generator
# import cv2
#
#
#
# class YoloDataset(Dataset):
#     def __init__(self, train_lines, image_size,  is_train=True):
#         super(YoloDataset, self).__init__()
#
#         self.train_lines = train_lines
#         self.train_batches = len(train_lines)
#         self.image_size = image_size
#         self.flag = True
#         self.is_train = is_train
#
#     def __len__(self):
#         return self.train_batches
#
#     def rand(self, a=0, b=1):
#         return np.random.rand() * (b - a) + a
#
#     def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
#         """实时数据增强的随机预处理"""
#         line = annotation_line.split()
#         # print("line: \n", line)
#         # print("line[0]: \n", line[0])
#         # print("line[1:]: \n", line[1:])
#         image = Image.open(line[0])
#         # print("image_size: \n", image.size)
#         # image.show() # 是数据集中的图片 扁平的 500x375
#
#
#         iw, ih = image.size  # 500 375
#         # print("iw, ih: \n", iw, ih)
#         h, w = input_shape # 416 416
#         # print("h, w: \n", h, w)
#
#         box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
#         # print("box: \n", box)
#         # print("box[1]: \n", box[1])
#
#         if not random: # 训练集不进来，测试集进来
#             scale = min(w/iw, h/ih)
#             nw = int(iw*scale)
#             nh = int(ih*scale)
#             dx = (w-nw)//2
#             dy = (h-nh)//2
#
#             image = image.resize((nw,nh), Image.BICUBIC)
#             # new_image = Image.new('RGB', (w,h), (128,128,128))
#             new_image = Image.new('L', (w, h), (128)) # 128代表灰色
#             # new_image.show()
#             # exit()
#             new_image.paste(image, (dx, dy))
#
#             # print(new_image, 'new_image')
#             # new_image.show()
#             image_data = np.array(new_image, np.float32)
#             # print(image_data)
#
#             # 调整目标框坐标
#             box_data = np.zeros((len(box), 5))
#             if len(box) > 0:
#                 np.random.shuffle(box)
#                 box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
#                 box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
#                 box[:, 0:2][box[:, 0:2] < 0] = 0
#                 box[:, 2][box[:, 2] > w] = w
#                 box[:, 3][box[:, 3] > h] = h
#                 box_w = box[:, 2] - box[:, 0]
#                 box_h = box[:, 3] - box[:, 1]
#                 box = box[np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框
#                 box_data = np.zeros((len(box), 5))
#                 box_data[:len(box)] = box
#
#             # print(box_data)
#             # exit()
#             return image_data, box_data
#
#         # 调整图片大小
#         new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
#         scale = self.rand(.25, 2)
#         # print("new_ar: \n", new_ar)
#         # print("scale: \n", scale)
#
#         if new_ar < 1:
#             nh = int(scale * h)
#             nw = int(nh * new_ar)
#         else:
#             nw = int(scale * w)
#             nh = int(nw / new_ar)
#         image = image.resize((nw, nh), Image.BICUBIC)
#
#         # print("image_size: \n", image.size)  #随机值
#         # image.show()
#         # print("nw: \n", nw)
#         # print("nh: \n", nh)
#
#
#
#         # 放置图片
#         dx = int(self.rand(0, w - nw))
#         dy = int(self.rand(0, h - nh))
#         # print("dx: \n", dx)
#         # print("dy: \n", dy)
#
#         new_image = Image.new(mode='L', size=(w, h),
#                               color=(np.random.randint(0, 255)))
#         # new_image.show() # 是一张size=416x416的灰色图片
#
#
#         new_image.paste(image, (dx, dy))
#         image = new_image
#         # print("image.size: \n", image.size)
#         # image.show()
#
#
#         # 是否翻转图片
#         flip = self.rand() < .5
#         if flip:
#             image = image.transpose(Image.FLIP_LEFT_RIGHT)
#         # print("image_size: \n",image.size)
#         # image.show()
#         # image.save("./aaa.bmp")
#
#
#         # # 色域变换
#         # hue = self.rand(-hue, hue)
#         # sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
#         # val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
#         # x = cv2.cvtColor(np.array(image, np.float32)/255, cv2.COLOR_RGB2HSV)
#         # x[..., 0] += hue*360
#         # x[..., 0][x[..., 0]>1] -= 1
#         # x[..., 0][x[..., 0]<0] += 1
#         # x[..., 1] *= sat
#         # x[..., 2] *= val
#         # x[x[:,:, 0]>360, 0] = 360
#         # x[:, :, 1:][x[:, :, 1:]>1] = 1
#         # x[x<0] = 0
#         # image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
#
#         # 调整目标框坐标
#         # print("len_box: \n", len(box))
#
#         box_data = np.zeros((len(box), 5))
#         # print("box_data: \n", box_data)
#
#         if len(box) > 0:
#             np.random.shuffle(box)
#             box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
#             box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
#             if flip:
#                 box[:, [0, 2]] = w - box[:, [2, 0]]
#             box[:, 0:2][box[:, 0:2] < 0] = 0
#             box[:, 2][box[:, 2] > w] = w
#             box[:, 3][box[:, 3] > h] = h
#             box_w = box[:, 2] - box[:, 0]
#             box_h = box[:, 3] - box[:, 1]
#             box = box[np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框
#             box_data = np.zeros((len(box), 5))
#             box_data[:len(box)] = box
#
#             # print("box_data: \n", box_data)  # 框的位置坐标正确
#
#         return image, box_data
#
#
#     def __getitem__(self, index):
#         lines = self.train_lines
#         n = self.train_batches  # 9
#         index = index % n  # 3
#         # print("index: \n", index)
#         # exit()
#         img, y = self.get_random_data(lines[index], self.image_size[0:2], random=self.is_train)
#         # img.show()
#         # print(img.size)
#         # img.save('./aaa.bmp') # 正确
#         # print(y, 'y') # 能和上一行对上
#         # exit()
#
#         # print(y[:, :4])
#         # exit()
#
#         if len(y) != 0:
#             # 从坐标转换成0~1的百分比
#             boxes = np.array(y[:, :4], dtype=np.float32)
#             boxes[:, 0] = boxes[:, 0] / self.image_size[1]  # /416
#             boxes[:, 1] = boxes[:, 1] / self.image_size[0]  # /416
#             boxes[:, 2] = boxes[:, 2] / self.image_size[1]
#             boxes[:, 3] = boxes[:, 3] / self.image_size[0]
#
#             boxes = np.maximum(np.minimum(boxes, 1), 0)  # 最小是0 最大是1
#             # print(boxes, 'boxes') # 到这里还是四个点的坐标呢，只不过是除以416之后的值
#             # exit()
#             boxes[:, 2] = boxes[:, 2] - boxes[:, 0] # 物体的宽W
#             boxes[:, 3] = boxes[:, 3] - boxes[:, 1] # 物体的高H
#
#             boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2 # 中心点坐标X
#             boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2 # 中心点坐标Y
#             # print(boxes, 'boxboxbox') # 真实框的中心点坐标、宽、高 归一化后
#
#             y = np.concatenate([boxes, y[:, -1:]], axis=-1)
#             # print(y, 'yyyy')  #这就是真实框的中心点坐标、宽、高 归一化 加上 类别 后的值
#             # exit()
#
#         img = np.array(img, dtype=np.float32)
#         #############自己加的扩充维度 (416x416 - > 1x416x416) #################
#         img = np.expand_dims(img, axis=0)
#         #############自己加的扩充维度 (416x416 - > 1x416x416) #################
#         # print(img, 'img')
#         # exit()
#         # tmp_inp = np.transpose(img / 255.0, (2, 0, 1))
#         # tmp_inp = np.transpose(img / 255.0)
#         tmp_inp = np.array(img / 255.0)
#         # print(tmp_inp, 'tmp_img')
#         tmp_targets = np.array(y, dtype=np.float32)
#
#         # print('tem_inp: \n', tmp_inp)
#         # print('tem_targets: \n', tmp_targets)
#         # print(tmp_inp.size)
#         # print(tmp_inp.shape)
#         # exit()
#
#         return tmp_inp, tmp_targets
#
#
# # DataLoader中collate_fn使用
# def yolo_dataset_collate(batch):
#     images = []
#     bboxes = []
#     for img, box in batch:
#         images.append(img)
#         bboxes.append(box)
#     images = np.array(images)
#     return images, bboxes
#


from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from utils.utils import bbox_iou, merge_bboxes
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from nets.yolo_training import Generator
import cv2

class YoloDataset(Dataset):
    def __init__(self, train_lines, image_size, mosaic=True, is_train=True):
        super(YoloDataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.mosaic = mosaic
        self.flag = True
        self.is_train = is_train

    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        """实时数据增强的随机预处理"""
        line = annotation_line.split()
        image = Image.open(line[0])
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            image = image.resize((nw,nh), Image.BICUBIC)
            # new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image = Image.new('L', (w, h), (128))
            new_image.paste(image, (dx, dy))

            # new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)

            image_data = np.array(new_image, np.float32)

            # 调整目标框坐标
            box_data = np.zeros((len(box), 5))
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            return image_data, box_data

        # 调整图片大小
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 放置图片
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        # new_image = Image.new('RGB', (w, h),
        #                       (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        new_image = Image.new(mode='L', size=(w, h), color=(np.random.randint(0, 255)))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 是否翻转图片
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 色域变换
        # hue = self.rand(-hue, hue)
        # sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        # val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        # x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        # x[..., 0] += hue*360
        # x[..., 0][x[..., 0]>1] -= 1
        # x[..., 0][x[..., 0]<0] += 1
        # x[..., 1] *= sat
        # x[..., 2] *= val
        # x[x[:,:, 0]>360, 0] = 360
        # x[:, :, 1:][x[:, :, 1:]>1] = 1
        # x[x<0] = 0
        # image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        # image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)


        # 调整目标框坐标
        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box

        return image, box_data

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, hue=.1, sat=1.5, val=1.5):
        h, w = input_shape
        min_offset_x = 0.3
        min_offset_y = 0.3
        scale_low = 1 - min(min_offset_x, min_offset_y)
        scale_high = scale_low + 0.2

        image_datas = []
        box_datas = []
        index = 0

        place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
        place_y = [0, int(h * min_offset_y), int(h * min_offset_y), 0]
        for line in annotation_line:
            # 每一行进行分割
            line_content = line.split()
            # 打开图片
            image = Image.open(line_content[0])

            image = image.convert("L")
            # 图片的大小
            iw, ih = image.size
            # 保存框的位置
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

            # 是否翻转图片
            flip = self.rand() < .5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            # 对输入进来的图片进行缩放
            new_ar = w / h
            scale = self.rand(scale_low, scale_high)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            # 进行色域变换
            # hue = self.rand(-hue, hue)
            # sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
            # val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
            # x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
            # x[..., 0] += hue*360
            # x[..., 0][x[..., 0]>1] -= 1
            # x[..., 0][x[..., 0]<0] += 1
            # x[..., 1] *= sat
            # x[..., 2] *= val
            # x[x[:,:, 0]>360, 0] = 360
            # x[:, :, 1:][x[:, :, 1:]>1] = 1
            # x[x<0] = 0
            # image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) # numpy array, 0 to 1

            # image = Image.fromarray((image * 255).astype(np.uint8))
            # 将图片进行放置，分别对应四张分割图片的位置
            dx = place_x[index]
            dy = place_y[index]
            # new_image = Image.new('RGB', (w, h),
            #                       (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
            new_image = Image.new('L', (w, h), (np.random.randint(0, 255)))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            # 对box进行重新处理
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)

        # 将图片分割，放在一起
        cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
        cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))


        # new_image = np.zeros([h, w, 3])
        new_image = np.zeros([h, w])

        new_image[:cuty, :cutx] = image_datas[0][:cuty, :cutx]
        new_image[cuty:, :cutx] = image_datas[1][cuty:, :cutx]
        new_image[cuty:, cutx:] = image_datas[2][cuty:, cutx:]
        new_image[:cuty, cutx:] = image_datas[3][:cuty, cutx:]


        # 对框进行进一步的处理
        new_boxes = np.array(merge_bboxes(box_datas, cutx, cuty))

        return new_image, new_boxes

    def __getitem__(self, index):
        lines = self.train_lines
        n = self.train_batches
        index = index % n
        if self.mosaic:
            if self.flag and (index + 4) < n:
                img, y = self.get_random_data_with_Mosaic(lines[index:index + 4], self.image_size[0:2])
            else:
                img, y = self.get_random_data(lines[index], self.image_size[0:2], random=self.is_train)
            self.flag = bool(1-self.flag)
        else:
            img, y = self.get_random_data(lines[index], self.image_size[0:2], random=self.is_train)

        if len(y) != 0:
            # 从坐标转换成0~1的百分比
            boxes = np.array(y[:, :4], dtype=np.float32)
            boxes[:, 0] = boxes[:, 0] / self.image_size[1]
            boxes[:, 1] = boxes[:, 1] / self.image_size[0]
            boxes[:, 2] = boxes[:, 2] / self.image_size[1]
            boxes[:, 3] = boxes[:, 3] / self.image_size[0]

            boxes = np.maximum(np.minimum(boxes, 1), 0)
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
            boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
            y = np.concatenate([boxes, y[:, -1:]], axis=-1)

        img = np.array(img, dtype=np.float32)
        #############自己加的扩充维度 (416x416 - > 1x416x416) #################
        img = np.expand_dims(img, axis=0)
        #############自己加的扩充维度 (416x416 - > 1x416x416) #################

        # tmp_inp = np.transpose(img / 255.0, (2, 0, 1))
        tmp_inp = np.array(img/255.0)
        tmp_targets = np.array(y, dtype=np.float32)
        return tmp_inp, tmp_targets


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes

