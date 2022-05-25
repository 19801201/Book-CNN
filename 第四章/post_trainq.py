# -*- coding: UTF-8 -*-
# encoding=utf-8
import os
import argparse
from torch.utils.data import DataLoader
import warnings
import numpy as np
from PIL import Image
import torch
from nets.yolo4_tiny_quan1 import YoloBody
import glob

warnings.filterwarnings("ignore")


def train():
    model_float_path = '0410weight/Epoch20-Total_Loss36.3260-Val_Loss0.0000.pth'
    model = YoloBody(3, 1)
    state_dict = torch.load(model_float_path)
    model.load_state_dict(state_dict)
    model.to("cpu")

    model.eval()
    model.fuse_model()
    print(model)

    # ENGINE = 'fbgemm'
    # torch.backends.quantized.engine = ENGINE
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)

    image_path = 'VOCdevkit/VOC2007/JPEGImages/'

    for epoch in range(0, 1):
        model.eval()
        for iter, img_path in enumerate(glob.glob(os.path.join(image_path, '*.jpg'))):
            image = Image.open(img_path)
            image_shape = np.array(np.shape(image)[0:2])
            crop_img = image.convert('L')
            crop_img = crop_img.resize((640, 640), Image.BICUBIC)
            photo = np.array(crop_img, dtype=np.float32) / 255.0
            # photo = np.transpose(photo, (2, 0, 1))
            photo = np.expand_dims(photo, axis=0)

            images = [photo]

            with torch.no_grad():
                images = torch.from_numpy(np.asarray(images))
                outputs = model(images)
            print(iter)

            if iter == 100000:
                break

        torch.quantization.convert(model, inplace=True)

        torch.jit.save(torch.jit.script(model),
                       "quanpth_jt/Epoch{}-YOLOV4_tiny_quantization_post.pth".format(epoch + 1))

        print("quantized=================================================")
        print(model)

        for k, v in model.state_dict().items():
            # print(k)
            # print(v)

            if not ('NoneType' in str(type(v))):
                if 'weight' in k:
                    np.save('./para800_1028_jt/' + k + '.scale', v.q_per_channel_scales())
                    np.save('./para800_1028_jt/' + k + '.zero_point', v.q_per_channel_zero_points())
                    np.save('./para800_1028_jt/' + k + '.int', v.int_repr())
                    np.save('./para800_1028_jt/' + k, v.dequantize().numpy())
                elif 'bias' in k:
                    np.save('./para800_1028_jt/' + k, v.detach().numpy())
                elif 'zero_point' in k:
                    np.save('./para800_1028_jt/' + k, v.detach().numpy())
                elif 'scale' in k:
                    np.save('./para800_1028_jt/' + k, v.detach().numpy())


if __name__ == "__main__":
    train()
