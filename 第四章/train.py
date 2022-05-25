import os
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utilsz.dataloader import yolo_dataset_collate, YoloDataset
from nets.yolo_training import YOLOLoss
from nets.yolo4_tiny_quan_split_leaky_relu_01 import YoloBody
from tqdm import tqdm
from collections import OrderedDict

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   获得类和先验框
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1,3,2])

def fit_one_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    total_loss = 0
    val_loss = 0

    net.train()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]

            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            # print(images.shape) # 16 1 416 416  bs c h w
            # exit()

            outputs = net(images)
            losses = []
            num_pos_all = 0
            #----------------------#
            #   计算损失
            #----------------------#
            for i in range(2):
                loss_item, num_pos = yolo_losses[i](outputs[i], targets)
                losses.append(loss_item)
                num_pos_all += num_pos

            loss = sum(losses) / num_pos_all
            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                outputs = net(images_val)
                losses = []
                num_pos_all = 0
                for i in range(2):
                    loss_item, num_pos = yolo_losses[i](outputs[i], targets_val)
                    losses.append(loss_item)
                    num_pos_all += num_pos
                loss = sum(losses) / num_pos_all
                val_loss += loss.item()
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), '0410weight/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))


#----------------------------------------------------#
#----------------------------------------------------#
if __name__ == "__main__":
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    Cuda = True
    #-------------------------------#
    #   Dataloder的使用
    #-------------------------------#
    Use_Data_Loader = True
    #------------------------------------------------------#
    #   是否对损失进行归一化，用于改变loss的大小
    #   用于决定计算最终loss是除上batch_size还是除上正样本数量
    #------------------------------------------------------#
    normalize = False
    #-------------------------------#
    #   输入的shape大小
    #   显存比较小可以使用416x416
    #   显存比较大可以使用608x608
    #-------------------------------#
    input_shape = (800, 800)

    #----------------------------------------------------#
    #   classes和anchor的路径，非常重要
    #   训练前一定要修改classes_path，使其对应自己的数据集
    #----------------------------------------------------#
    anchors_path = 'model_data/yolo_anchors.txt'
    classes_path = 'model_data/new_classes.txt'
    #----------------------------------------------------#
    #   获取classes和anchor
    #----------------------------------------------------#
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names) #1

    
    #------------------------------------------------------#
    #   Yolov4的tricks应用
    #   mosaic 马赛克数据增强 True or False 
    #   实际测试时mosaic数据增强并不稳定，所以默认为False
    #   Cosine_scheduler 余弦退火学习率 True or False
    #   label_smoothing 标签平滑
    #------------------------------------------------------#
    mosaic = False
    Cosine_lr = False
    smoooth_label = False
    
    #------------------------------------------------------#
    #   创建yolo模型
    #   训练前一定要修改classes_path和对应的txt文件
    #------------------------------------------------------#
    model = YoloBody(len(anchors[0]), num_classes)

    #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #------------------------------------------------------#
    # model_path = "model_data/Epoch99-Total_Loss4.0353-Val_Loss4.1590.pth"
    # print('Loading weights into state dict...')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_dict = model.state_dict()
    #
    # # a = list(model_dict.values())[116]
    # # print(a)
    # pretrained_dict = torch.load(model_path, map_location=device)

    # b = list(pretrained_dict.values())[116]
    # print(b)
    # d1 = OrderedDict([('conv1.conv.0.weight', v) if k == 'backbone.conv1.conv.weight' else (k, v) for k, v in
    #                   pretrained_dict.items()])
    # d2 = OrderedDict([('conv1.conv.1.weight', v) if k == 'backbone.conv1.bn.weight' else (k, v) for k, v in d1.items()])
    # d3 = OrderedDict([('conv1.conv.1.bias', v) if k == 'backbone.conv1.bn.bias' else (k, v) for k, v in d2.items()])
    # d4 = OrderedDict(
    #     [('conv1.conv.1.running_mean', v) if k == 'backbone.conv1.bn.running_mean' else (k, v) for k, v in d3.items()])
    # d5 = OrderedDict(
    #     [('conv1.conv.1.running_var', v) if k == 'backbone.conv1.bn.running_var' else (k, v) for k, v in d4.items()])
    # d6 = OrderedDict(
    #     [('conv1.conv.1.num_batches_tracked', v) if k == 'backbone.conv1.bn.num_batches_tracked' else (k, v) for k, v in
    #      d5.items()])
    # d7 = OrderedDict(
    #     [('conv2.conv.0.weight', v) if k == 'backbone.conv2.conv.weight' else (k, v) for k, v in d6.items()])
    # d8 = OrderedDict([('conv2.conv.1.weight', v) if k == 'backbone.conv2.bn.weight' else (k, v) for k, v in d7.items()])
    # d9 = OrderedDict([('conv2.conv.1.bias', v) if k == 'backbone.conv2.bn.bias' else (k, v) for k, v in d8.items()])
    # d10 = OrderedDict(
    #     [('conv2.conv.1.running_mean', v) if k == 'backbone.conv2.bn.running_mean' else (k, v) for k, v in d9.items()])
    # d11 = OrderedDict(
    #     [('conv2.conv.1.running_var', v) if k == 'backbone.conv2.bn.running_var' else (k, v) for k, v in d10.items()])
    # d12 = OrderedDict(
    #     [('conv2.conv.1.num_batches_tracked', v) if k == 'backbone.conv2.bn.num_batches_tracked' else (k, v) for k, v in
    #      d11.items()])
    # d13 = OrderedDict(
    #     [('resblock_body1.conv1.conv.0.weight', v) if k == 'backbone.resblock_body1.conv1.conv.weight' else (k, v) for
    #      k, v in d12.items()])
    # d14 = OrderedDict(
    #     [('resblock_body1.conv1.conv.1.weight', v) if k == 'backbone.resblock_body1.conv1.bn.weight' else (k, v) for
    #      k, v in d13.items()])
    # d15 = OrderedDict(
    #     [('resblock_body1.conv1.conv.1.bias', v) if k == 'backbone.resblock_body1.conv1.bn.bias' else (k, v) for k, v in
    #      d14.items()])
    # d16 = OrderedDict([('resblock_body1.conv1.conv.1.running_mean',
    #                     v) if k == 'backbone.resblock_body1.conv1.bn.running_mean' else (k, v) for k, v in d15.items()])
    # d17 = OrderedDict([('resblock_body1.conv1.conv.1.running_var',
    #                     v) if k == 'backbone.resblock_body1.conv1.bn.running_var' else (k, v) for k, v in d16.items()])
    # d18 = OrderedDict([('resblock_body1.conv1.conv.1.num_batches_tracked',
    #                     v) if k == 'backbone.resblock_body1.conv1.bn.num_batches_tracked' else (k, v) for k, v in
    #                    d17.items()])
    # d19 = OrderedDict(
    #     [('resblock_body1.conv2.conv.0.weight', v) if k == 'backbone.resblock_body1.conv2.conv.weight' else (k, v) for
    #      k, v in d18.items()])
    # d20 = OrderedDict(
    #     [('resblock_body1.conv2.conv.1.weight', v) if k == 'backbone.resblock_body1.conv2.bn.weight' else (k, v) for
    #      k, v in d19.items()])
    # d21 = OrderedDict(
    #     [('resblock_body1.conv2.conv.1.bias', v) if k == 'backbone.resblock_body1.conv2.bn.bias' else (k, v) for k, v in
    #      d20.items()])
    # d22 = OrderedDict([('resblock_body1.conv2.conv.1.running_mean',
    #                     v) if k == 'backbone.resblock_body1.conv2.bn.running_mean' else (k, v) for k, v in d21.items()])
    # d23 = OrderedDict([('resblock_body1.conv2.conv.1.running_var',
    #                     v) if k == 'backbone.resblock_body1.conv2.bn.running_var' else (k, v) for k, v in d22.items()])
    # d24 = OrderedDict([('resblock_body1.conv2.conv.1.num_batches_tracked',
    #                     v) if k == 'backbone.resblock_body1.conv2.bn.num_batches_tracked' else (k, v) for k, v in
    #                    d23.items()])
    # d25 = OrderedDict(
    #     [('resblock_body1.conv3.conv.0.weight', v) if k == 'backbone.resblock_body1.conv3.conv.weight' else (k, v) for
    #      k, v in d24.items()])
    # d26 = OrderedDict(
    #     [('resblock_body1.conv3.conv.1.weight', v) if k == 'backbone.resblock_body1.conv3.bn.weight' else (k, v) for
    #      k, v in d25.items()])
    # d27 = OrderedDict(
    #     [('resblock_body1.conv3.conv.1.bias', v) if k == 'backbone.resblock_body1.conv3.bn.bias' else (k, v) for k, v in
    #      d26.items()])
    # d28 = OrderedDict([('resblock_body1.conv3.conv.1.running_mean',
    #                     v) if k == 'backbone.resblock_body1.conv3.bn.running_mean' else (k, v) for k, v in d27.items()])
    # d29 = OrderedDict([('resblock_body1.conv3.conv.1.running_var',
    #                     v) if k == 'backbone.resblock_body1.conv3.bn.running_var' else (k, v) for k, v in d28.items()])
    # d30 = OrderedDict([('resblock_body1.conv3.conv.1.num_batches_tracked',
    #                     v) if k == 'backbone.resblock_body1.conv3.bn.num_batches_tracked' else (k, v) for k, v in
    #                    d29.items()])
    # d31 = OrderedDict(
    #     [('resblock_body1.conv4.conv.0.weight', v) if k == 'backbone.resblock_body1.conv4.conv.weight' else (k, v) for
    #      k, v in d30.items()])
    # d32 = OrderedDict(
    #     [('resblock_body1.conv4.conv.1.weight', v) if k == 'backbone.resblock_body1.conv4.bn.weight' else (k, v) for
    #      k, v in d31.items()])
    # d33 = OrderedDict(
    #     [('resblock_body1.conv4.conv.1.bias', v) if k == 'backbone.resblock_body1.conv4.bn.bias' else (k, v) for k, v in
    #      d32.items()])
    # d34 = OrderedDict([('resblock_body1.conv4.conv.1.running_mean',
    #                     v) if k == 'backbone.resblock_body1.conv4.bn.running_mean' else (k, v) for k, v in d33.items()])
    # d35 = OrderedDict([('resblock_body1.conv4.conv.1.running_var',
    #                     v) if k == 'backbone.resblock_body1.conv4.bn.running_var' else (k, v) for k, v in d34.items()])
    # d36 = OrderedDict([('resblock_body1.conv4.conv.1.num_batches_tracked',
    #                     v) if k == 'backbone.resblock_body1.conv4.bn.num_batches_tracked' else (k, v) for k, v in
    #                    d35.items()])
    # d37 = OrderedDict(
    #     [('resblock_body2.conv1.conv.0.weight', v) if k == 'backbone.resblock_body2.conv1.conv.weight' else (k, v) for
    #      k, v in d36.items()])
    # d38 = OrderedDict(
    #     [('resblock_body2.conv1.conv.1.weight', v) if k == 'backbone.resblock_body2.conv1.bn.weight' else (k, v) for
    #      k, v in d37.items()])
    # d39 = OrderedDict(
    #     [('resblock_body2.conv1.conv.1.bias', v) if k == 'backbone.resblock_body2.conv1.bn.bias' else (k, v) for k, v in
    #      d38.items()])
    # d40 = OrderedDict([('resblock_body2.conv1.conv.1.running_mean',
    #                     v) if k == 'backbone.resblock_body2.conv1.bn.running_mean' else (k, v) for k, v in d39.items()])
    # d41 = OrderedDict([('resblock_body2.conv1.conv.1.running_var',
    #                     v) if k == 'backbone.resblock_body2.conv1.bn.running_var' else (k, v) for k, v in d40.items()])
    # d42 = OrderedDict([('resblock_body2.conv1.conv.1.num_batches_tracked',
    #                     v) if k == 'backbone.resblock_body2.conv1.bn.num_batches_tracked' else (k, v) for k, v in
    #                    d41.items()])
    # d43 = OrderedDict(
    #     [('resblock_body2.conv2.conv.0.weight', v) if k == 'backbone.resblock_body2.conv2.conv.weight' else (k, v) for
    #      k, v in d42.items()])
    # d44 = OrderedDict(
    #     [('resblock_body2.conv2.conv.1.weight', v) if k == 'backbone.resblock_body2.conv2.bn.weight' else (k, v) for
    #      k, v in d43.items()])
    # d45 = OrderedDict(
    #     [('resblock_body2.conv2.conv.1.bias', v) if k == 'backbone.resblock_body2.conv2.bn.bias' else (k, v) for k, v in
    #      d44.items()])
    # d46 = OrderedDict([('resblock_body2.conv2.conv.1.running_mean',
    #                     v) if k == 'backbone.resblock_body2.conv2.bn.running_mean' else (k, v) for k, v in d45.items()])
    # d47 = OrderedDict([('resblock_body2.conv2.conv.1.running_var',
    #                     v) if k == 'backbone.resblock_body2.conv2.bn.running_var' else (k, v) for k, v in d46.items()])
    # d48 = OrderedDict([('resblock_body2.conv2.conv.1.num_batches_tracked',
    #                     v) if k == 'backbone.resblock_body2.conv2.bn.num_batches_tracked' else (k, v) for k, v in
    #                    d47.items()])
    # d49 = OrderedDict(
    #     [('resblock_body2.conv3.conv.0.weight', v) if k == 'backbone.resblock_body2.conv3.conv.weight' else (k, v) for
    #      k, v in d48.items()])
    # d50 = OrderedDict(
    #     [('resblock_body2.conv3.conv.1.weight', v) if k == 'backbone.resblock_body2.conv3.bn.weight' else (k, v) for
    #      k, v in d49.items()])
    # d51 = OrderedDict(
    #     [('resblock_body2.conv3.conv.1.bias', v) if k == 'backbone.resblock_body2.conv3.bn.bias' else (k, v) for k, v in
    #      d50.items()])
    # d52 = OrderedDict([('resblock_body2.conv3.conv.1.running_mean',
    #                     v) if k == 'backbone.resblock_body2.conv3.bn.running_mean' else (k, v) for k, v in d51.items()])
    # d53 = OrderedDict([('resblock_body2.conv3.conv.1.running_var',
    #                     v) if k == 'backbone.resblock_body2.conv3.bn.running_var' else (k, v) for k, v in d52.items()])
    # d54 = OrderedDict([('resblock_body2.conv3.conv.1.num_batches_tracked',
    #                     v) if k == 'backbone.resblock_body2.conv3.bn.num_batches_tracked' else (k, v) for k, v in
    #                    d53.items()])
    # d55 = OrderedDict(
    #     [('resblock_body2.conv4.conv.0.weight', v) if k == 'backbone.resblock_body2.conv4.conv.weight' else (k, v) for
    #      k, v in d54.items()])
    # d56 = OrderedDict(
    #     [('resblock_body2.conv4.conv.1.weight', v) if k == 'backbone.resblock_body2.conv4.bn.weight' else (k, v) for
    #      k, v in d55.items()])
    # d57 = OrderedDict(
    #     [('resblock_body2.conv4.conv.1.bias', v) if k == 'backbone.resblock_body2.conv4.bn.bias' else (k, v) for k, v in
    #      d56.items()])
    # d58 = OrderedDict([('resblock_body2.conv4.conv.1.running_mean',
    #                     v) if k == 'backbone.resblock_body2.conv4.bn.running_mean' else (k, v) for k, v in d57.items()])
    # d59 = OrderedDict([('resblock_body2.conv4.conv.1.running_var',
    #                     v) if k == 'backbone.resblock_body2.conv4.bn.running_var' else (k, v) for k, v in d58.items()])
    # d60 = OrderedDict([('resblock_body2.conv4.conv.1.num_batches_tracked',
    #                     v) if k == 'backbone.resblock_body2.conv4.bn.num_batches_tracked' else (k, v) for k, v in
    #                    d59.items()])
    # d61 = OrderedDict(
    #     [('resblock_body3.conv1.conv.0.weight', v) if k == 'backbone.resblock_body3.conv1.conv.weight' else (k, v) for
    #      k, v in d60.items()])
    # d62 = OrderedDict(
    #     [('resblock_body3.conv1.conv.1.weight', v) if k == 'backbone.resblock_body3.conv1.bn.weight' else (k, v) for
    #      k, v in d61.items()])
    # d63 = OrderedDict(
    #     [('resblock_body3.conv1.conv.1.bias', v) if k == 'backbone.resblock_body3.conv1.bn.bias' else (k, v) for k, v in
    #      d62.items()])
    # d64 = OrderedDict([('resblock_body3.conv1.conv.1.running_mean',
    #                     v) if k == 'backbone.resblock_body3.conv1.bn.running_mean' else (k, v) for k, v in d63.items()])
    # d65 = OrderedDict([('resblock_body3.conv1.conv.1.running_var',
    #                     v) if k == 'backbone.resblock_body3.conv1.bn.running_var' else (k, v) for k, v in d64.items()])
    # d66 = OrderedDict([('resblock_body3.conv1.conv.1.num_batches_tracked',
    #                     v) if k == 'backbone.resblock_body3.conv1.bn.num_batches_tracked' else (k, v) for k, v in
    #                    d65.items()])
    # d67 = OrderedDict(
    #     [('resblock_body3.conv2.conv.0.weight', v) if k == 'backbone.resblock_body3.conv2.conv.weight' else (k, v) for
    #      k, v in d66.items()])
    # d68 = OrderedDict(
    #     [('resblock_body3.conv2.conv.1.weight', v) if k == 'backbone.resblock_body3.conv2.bn.weight' else (k, v) for
    #      k, v in d67.items()])
    # d69 = OrderedDict(
    #     [('resblock_body3.conv2.conv.1.bias', v) if k == 'backbone.resblock_body3.conv2.bn.bias' else (k, v) for k, v in
    #      d68.items()])
    # d70 = OrderedDict([('resblock_body3.conv2.conv.1.running_mean',
    #                     v) if k == 'backbone.resblock_body3.conv2.bn.running_mean' else (k, v) for k, v in d69.items()])
    # d71 = OrderedDict([('resblock_body3.conv2.conv.1.running_var',
    #                     v) if k == 'backbone.resblock_body3.conv2.bn.running_var' else (k, v) for k, v in d70.items()])
    # d72 = OrderedDict([('resblock_body3.conv2.conv.1.num_batches_tracked',
    #                     v) if k == 'backbone.resblock_body3.conv2.bn.num_batches_tracked' else (k, v) for k, v in
    #                    d71.items()])
    # d73 = OrderedDict(
    #     [('resblock_body3.conv3.conv.0.weight', v) if k == 'backbone.resblock_body3.conv3.conv.weight' else (k, v) for
    #      k, v in d72.items()])
    # d74 = OrderedDict(
    #     [('resblock_body3.conv3.conv.1.weight', v) if k == 'backbone.resblock_body3.conv3.bn.weight' else (k, v) for
    #      k, v in d73.items()])
    # d75 = OrderedDict(
    #     [('resblock_body3.conv3.conv.1.bias', v) if k == 'backbone.resblock_body3.conv3.bn.bias' else (k, v) for k, v in
    #      d74.items()])
    # d76 = OrderedDict([('resblock_body3.conv3.conv.1.running_mean',
    #                     v) if k == 'backbone.resblock_body3.conv3.bn.running_mean' else (k, v) for k, v in d75.items()])
    # d77 = OrderedDict([('resblock_body3.conv3.conv.1.running_var',
    #                     v) if k == 'backbone.resblock_body3.conv3.bn.running_var' else (k, v) for k, v in d76.items()])
    # d78 = OrderedDict([('resblock_body3.conv3.conv.1.num_batches_tracked',
    #                     v) if k == 'backbone.resblock_body3.conv3.bn.num_batches_tracked' else (k, v) for k, v in
    #                    d77.items()])
    # d79 = OrderedDict(
    #     [('resblock_body3.conv4.conv.0.weight', v) if k == 'backbone.resblock_body3.conv4.conv.weight' else (k, v) for
    #      k, v in d78.items()])
    # d80 = OrderedDict(
    #     [('resblock_body3.conv4.conv.1.weight', v) if k == 'backbone.resblock_body3.conv4.bn.weight' else (k, v) for
    #      k, v in d79.items()])
    # d81 = OrderedDict(
    #     [('resblock_body3.conv4.conv.1.bias', v) if k == 'backbone.resblock_body3.conv4.bn.bias' else (k, v) for k, v in
    #      d80.items()])
    # d82 = OrderedDict([('resblock_body3.conv4.conv.1.running_mean',
    #                     v) if k == 'backbone.resblock_body3.conv4.bn.running_mean' else (k, v) for k, v in d81.items()])
    # d83 = OrderedDict([('resblock_body3.conv4.conv.1.running_var',
    #                     v) if k == 'backbone.resblock_body3.conv4.bn.running_var' else (k, v) for k, v in d82.items()])
    # d84 = OrderedDict([('resblock_body3.conv4.conv.1.num_batches_tracked',
    #                     v) if k == 'backbone.resblock_body3.conv4.bn.num_batches_tracked' else (k, v) for k, v in
    #                    d83.items()])
    # d85 = OrderedDict(
    #     [('conv3.conv.0.weight', v) if k == 'backbone.conv3.conv.weight' else (k, v) for k, v in d84.items()])
    # d86 = OrderedDict(
    #     [('conv3.conv.1.weight', v) if k == 'backbone.conv3.bn.weight' else (k, v) for k, v in d85.items()])
    # d87 = OrderedDict([('conv3.conv.1.bias', v) if k == 'backbone.conv3.bn.bias' else (k, v) for k, v in d86.items()])
    # d88 = OrderedDict(
    #     [('conv3.conv.1.running_mean', v) if k == 'backbone.conv3.bn.running_mean' else (k, v) for k, v in d87.items()])
    # d89 = OrderedDict(
    #     [('conv3.conv.1.running_var', v) if k == 'backbone.conv3.bn.running_var' else (k, v) for k, v in d88.items()])
    # d90 = OrderedDict(
    #     [('conv3.conv.1.num_batches_tracked', v) if k == 'backbone.conv3.bn.num_batches_tracked' else (k, v) for k, v in
    #      d89.items()])
    # d91 = OrderedDict(
    #     [('conv_for_P5.conv.0.weight', v) if k == 'conv_for_P5.conv.weight' else (k, v) for k, v in d90.items()])
    # d92 = OrderedDict(
    #     [('conv_for_P5.conv.1.weight', v) if k == 'conv_for_P5.bn.weight' else (k, v) for k, v in d91.items()])
    # d93 = OrderedDict(
    #     [('conv_for_P5.conv.1.bias', v) if k == 'conv_for_P5.bn.bias' else (k, v) for k, v in d92.items()])
    # d94 = OrderedDict(
    #     [('conv_for_P5.conv.1.running_mean', v) if k == 'conv_for_P5.bn.running_mean' else (k, v) for k, v in
    #      d93.items()])
    # d95 = OrderedDict(
    #     [('conv_for_P5.conv.1.running_var', v) if k == 'conv_for_P5.bn.running_var' else (k, v) for k, v in
    #      d94.items()])
    # d96 = OrderedDict(
    #     [('conv_for_P5.conv.1.num_batches_tracked', v) if k == 'conv_for_P5.bn.num_batches_tracked' else (k, v) for k, v
    #      in d95.items()])
    # d97 = OrderedDict(
    #     [('yolo_headP5.0.conv.0.weight', v) if k == 'yolo_headP5.0.conv.weight' else (k, v) for k, v in d96.items()])
    # d98 = OrderedDict(
    #     [('yolo_headP5.0.conv.1.weight', v) if k == 'yolo_headP5.0.bn.weight' else (k, v) for k, v in d97.items()])
    # d99 = OrderedDict(
    #     [('yolo_headP5.0.conv.1.bias', v) if k == 'yolo_headP5.0.bn.bias' else (k, v) for k, v in d98.items()])
    # d100 = OrderedDict(
    #     [('yolo_headP5.0.conv.1.running_mean', v) if k == 'yolo_headP5.0.bn.running_mean' else (k, v) for k, v in
    #      d99.items()])
    # d101 = OrderedDict(
    #     [('yolo_headP5.0.conv.1.running_var', v) if k == 'yolo_headP5.0.bn.running_var' else (k, v) for k, v in
    #      d100.items()])
    # d102 = OrderedDict(
    #     [('yolo_headP5.0.conv.1.num_batches_tracked', v) if k == 'yolo_headP5.0.bn.num_batches_tracked' else (k, v) for
    #      k, v in d101.items()])
    # d103 = OrderedDict(
    #     [('yolo_headP5.1.weight', v) if k == 'yolo_headP5.1.weight' else (k, v) for k, v in d102.items()])
    # d104 = OrderedDict([('yolo_headP5.1.bias', v) if k == 'yolo_headP5.1.bias' else (k, v) for k, v in d103.items()])
    # d105 = OrderedDict(
    #     [('upsample.upsample.0.conv.0.weight', v) if k == 'upsample.upsample.0.conv.weight' else (k, v) for k, v in
    #      d104.items()])
    # d106 = OrderedDict(
    #     [('upsample.upsample.0.conv.1.weight', v) if k == 'upsample.upsample.0.bn.weight' else (k, v) for k, v in
    #      d105.items()])
    # d107 = OrderedDict(
    #     [('upsample.upsample.0.conv.1.bias', v) if k == 'upsample.upsample.0.bn.bias' else (k, v) for k, v in
    #      d106.items()])
    # d108 = OrderedDict(
    #     [('upsample.upsample.0.conv.1.running_mean', v) if k == 'upsample.upsample.0.bn.running_mean' else (k, v) for
    #      k, v in d107.items()])
    # d109 = OrderedDict(
    #     [('upsample.upsample.0.conv.1.running_var', v) if k == 'upsample.upsample.0.bn.running_var' else (k, v) for k, v
    #      in d108.items()])
    # d110 = OrderedDict([('upsample.upsample.0.conv.1.num_batches_tracked',
    #                      v) if k == 'upsample.upsample.0.bn.num_batches_tracked' else (k, v) for k, v in d109.items()])
    # d111 = OrderedDict(
    #     [('yolo_headP4.0.conv.0.weight', v) if k == 'yolo_headP4.0.conv.weight' else (k, v) for k, v in d110.items()])
    # d112 = OrderedDict(
    #     [('yolo_headP4.0.conv.1.weight', v) if k == 'yolo_headP4.0.bn.weight' else (k, v) for k, v in d111.items()])
    # d113 = OrderedDict(
    #     [('yolo_headP4.0.conv.1.bias', v) if k == 'yolo_headP4.0.bn.bias' else (k, v) for k, v in d112.items()])
    # d114 = OrderedDict(
    #     [('yolo_headP4.0.conv.1.running_mean', v) if k == 'yolo_headP4.0.bn.running_mean' else (k, v) for k, v in
    #      d113.items()])
    # d115 = OrderedDict(
    #     [('yolo_headP4.0.conv.1.running_var', v) if k == 'yolo_headP4.0.bn.running_var' else (k, v) for k, v in
    #      d114.items()])
    # d116 = OrderedDict(
    #     [('yolo_headP4.0.conv.1.num_batches_tracked', v) if k == 'yolo_headP4.0.bn.num_batches_tracked' else (k, v) for
    #      k, v in d115.items()])
    # d117 = OrderedDict(
    #     [('yolo_headP4.1.weight', v) if k == 'yolo_headP4.1.weight' else (k, v) for k, v in d116.items()])
    # d118 = OrderedDict([('yolo_headP4.1.bias', v) if k == 'yolo_headP4.1.bias' else (k, v) for k, v in d117.items()])

    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}

    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    # model_dict.update(pretrained_dict)
    # c = list(model_dict.values())[116]
    # print(c)
    # print('=====================================================')
    # print(a.shape)
    # print(b.shape)
    # print(c.shape)
    # print('equal?',c.cuda().equal(b))

    # model.load_state_dict(model_dict)

    # for k,v in model.named_parameters():
    #     if k == 'conv1.conv.0.weight':
    #         nn.init.normal_(v, mean=0, std=0.01)

    print('Finished!')

    net = model.train()

    if Cuda:
        net = net.cuda()


    # 建立loss函数
    yolo_losses = []
    for i in range(2):  # 0 1
        yolo_losses.append(YOLOLoss(np.reshape(anchors,[-1,2]),num_classes, \
                                (input_shape[1], input_shape[0]), smoooth_label, Cuda, normalize))


    #----------------------------------------------------#
    #   获得图片路径和标签
    #----------------------------------------------------#
    annotation_path = '2007_train.txt'
    #----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    #----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        lr = 1e-3
        Batch_size = 16
        Init_Epoch = 0
        Freeze_Epoch = 300
        
        #----------------------------------------------------------------------------#
        #   我在实际测试时，发现optimizer的weight_decay起到了反作用，
        #   所以去除掉了weight_decay，大家也可以开起来试试，一般是weight_decay=5e-4
        #----------------------------------------------------------------------------#
        optimizer = optim.Adam(net.parameters(),lr)
        #optimizer = optim.SGD(net.parameters(), lr, momentum=0.9)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=15,gamma=0.8)


        if Use_Data_Loader:
            train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), is_train=True)

            val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), is_train=False)


            gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)

            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)


        epoch_size = max(1, num_train//Batch_size)
        epoch_size_val = num_val//Batch_size
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        """
        for param in model.quant.parameters():
            param.requires_grad = False
            
        for param in model.conv1.parameters():
            param.requires_grad = False
            
        for param in model.conv2.parameters():
            param.requires_grad = False
            
        for param in model.resblock_body1.parameters():
            param.requires_grad = False
            
        for param in model.resblock_body2.parameters():
            param.requires_grad = False
            
        for param in model.resblock_body3.parameters():
            param.requires_grad = False
            
        for param in model.conv3.parameters():
            param.requires_grad = False
        """    
        
        # for param in model.backbone.parameters():
        #     param.requires_grad = False
        # for key,params in model.named_parameters():
        #     if key != 'conv1.conv.0.weight':
        #         params.requires_grad = False

        for epoch in range(Init_Epoch,Freeze_Epoch):
            
            if epoch >= 100 and epoch <= 140:
                for param in model.quant.parameters():
                    param.requires_grad = False
                for param in model.conv1.parameters():
                    param.requires_grad = False            
                for param in model.conv2.parameters():
                    param.requires_grad = False            
                for param in model.resblock_body1.parameters():
                    param.requires_grad = False            
                for param in model.resblock_body2.parameters():
                    param.requires_grad = False            
                for param in model.resblock_body3.parameters():
                    param.requires_grad = False            
                for param in model.conv3.parameters():
                    param.requires_grad = False 
                
            else :
                for param in model.quant.parameters():
                    param.requires_grad = True
                for param in model.conv1.parameters():
                    param.requires_grad = True            
                for param in model.conv2.parameters():
                    param.requires_grad = True            
                for param in model.resblock_body1.parameters():
                    param.requires_grad = True            
                for param in model.resblock_body2.parameters():
                    param.requires_grad = True            
                for param in model.resblock_body3.parameters():
                    param.requires_grad = True            
                for param in model.conv3.parameters():
                    param.requires_grad = True
            
       
            fit_one_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda)
            lr_scheduler.step()
            
    
    """        
    if True:
        lr = 1e-4
        Batch_size = 64
        Init_Epoch = 80
        Freeze_Epoch = 120
        
        #----------------------------------------------------------------------------#
        #   我在实际测试时，发现optimizer的weight_decay起到了反作用，
        #   所以去除掉了weight_decay，大家也可以开起来试试，一般是weight_decay=5e-4
        #----------------------------------------------------------------------------#
        # optimizer = optim.Adam(net.parameters(),lr)
        optimizer = optim.SGD(net.parameters(), lr, momentum=0.9)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.92)


        if Use_Data_Loader:
            train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), is_train=True)

            val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), is_train=False)


            gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)

            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)



        epoch_size = max(1, num_train//Batch_size)
        epoch_size_val = num_val//Batch_size
        #------------------------------------#
        #   解冻一定部分训练
        #------------------------------------#
        for param in model.quant.parameters():
            param.requires_grad = True
            
        for param in model.conv1.parameters():
            param.requires_grad = True
            
        for param in model.conv2.parameters():
            param.requires_grad = True
            
        for param in model.resblock_body1.parameters():
            param.requires_grad = True
            
        for param in model.resblock_body2.parameters():
            param.requires_grad = True
            
        for param in model.resblock_body3.parameters():
            param.requires_grad = True
            
        for param in model.conv3.parameters():
            param.requires_grad = True
        # for param in model.backbone.parameters():
        #     param.requires_grad = False
        # for key,params in model.named_parameters():
        #     if key != 'conv1.conv.0.weight':
        #         params.requires_grad = False

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_one_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda)
            lr_scheduler.step()
    """
    
    
    
    

    # if True:
    #     lr = 1e-4
    #     Batch_size = 16
    #     Init_Epoch = 10
    #     Freeze_Epoch = 100
    #
    #     # ----------------------------------------------------------------------------#
    #     #   我在实际测试时，发现optimizer的weight_decay起到了反作用，
    #     #   所以去除掉了weight_decay，大家也可以开起来试试，一般是weight_decay=5e-4
    #     # ----------------------------------------------------------------------------#
    #     optimizer = optim.Adam(net.parameters(), lr)
    #     if Cosine_lr:
    #         lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    #     else:
    #         lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
    #
    #     if Use_Data_Loader:
    #         train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), is_train=True)
    #
    #         val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), is_train=False)
    #
    #         gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
    #                          drop_last=True, collate_fn=yolo_dataset_collate)
    #
    #         gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
    #                              drop_last=True, collate_fn=yolo_dataset_collate)
    #
    #     epoch_size = max(1, num_train // Batch_size)
    #     epoch_size_val = num_val // Batch_size
    #     # ------------------------------------#
    #     #   解冻一定部分训练
    #     # ------------------------------------#
    #     # for param in model.backbone.parameters():
    #     #     param.requires_grad = False
    #     for key, params in model.named_parameters():
    #         if key != 'conv1.conv.0.weight':
    #             params.requires_grad = True
    #
    #     for epoch in range(Init_Epoch, Freeze_Epoch):
    #         fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, Cuda)
    #         lr_scheduler.step()
