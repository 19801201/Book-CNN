#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#   map测试请看get_dr_txt.py、get_gt_txt.py
#   和get_map.py
#--------------------------------------------#
# import torch
from torchsummary import summary
from nets.yolo4_tiny_quan1 import YoloBody

if __name__ == "__main__":
    # import torch
    # # 需要使用device来指定网络在GPU还是CPU运行
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = YoloBody(3, 1).to(device)
    # summary(model, input_size=(1, 640, 640))

    from torchstat import stat
    from nets.yolo4_tiny_quan1 import YoloBody
    import time
    import numpy as np
    model = YoloBody(3, 1)
    start = time.time()
    stat(model, (1, 640, 640))
    end = time.time()
    print("time:", end-start)