# import datetime
# import paddle
# import paddle.nn.functional as F
# from paddle.io import DataLoader
# from lib import dataset
# import numpy as np
# import cv2
# import argparse
# import os
# import random
# from visualdl import LogWriter
# bs = 128
import datetime
import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader
from lib import dataset
import numpy as np
import cv2
import argparse
import os
import random

# def config():
#     parser = argparse.ArgumentParser(description='train params')
#     parser.add_argument('--Min_LR', default=0.00016, help='min lr')
#     parser.add_argument('--Max_LR', default=0.016)
#     parser.add_argument('--top_epoch', default=5)
#     parser.add_argument('--epoch', default=69)
#     parser.add_argument('--mode_path', default=False, help='where your pretrained model')
#     parser.add_argument('--train_bs', default=16)
#     parser.add_argument('--test_bs', default=8)
#     parser.add_argument('--text_step', default=int(10553 // 16 // 2), help='if step % text_step == 0 eval')
#     parser.add_argument('--min_mae', default=10)
#     parser.add_argument('--show_step', default=30)
#     parser.add_argument('--eval_dataset', default=r'work/PASCALS')
#     parser.add_argument('--train_dataset', default=r'work/DUTS-TR')
#     parser.add_argument('--save_path', default='res_train_weight/fmf_dila2_fm1')
#     parser.add_argument('--save_iter', default=1, help=r'every iter to save model')
#     parser.add_argument('--log_dir', default='./log/weight_m3_0.5')
#     cag = parser.parse_args()
#     return cag


# cag = config()
def config():
    parser = argparse.ArgumentParser(description='train params')
    parser.add_argument('--Min_LR', default=0.000016, help='min lr')
    parser.add_argument('--Max_LR', default=0.016)
    parser.add_argument('--top_epoch', default=5)
    parser.add_argument('--epoch', default=64)
    parser.add_argument('--mode_path', default=False, help='where your pretrained model')
    parser.add_argument('--train_bs', default=16)
    parser.add_argument('--test_bs', default=16)
    parser.add_argument('--text_step', default=int(10553 // 16 // 2), help='if step % text_step == 0 eval')
    parser.add_argument('--min_mae', default=10)
    parser.add_argument('--show_step', default=50)
    parser.add_argument('--eval_dataset', default=r'work/PASCALS')
    parser.add_argument('--train_dataset', default=r'work/DUTS-TR')
    parser.add_argument('--save_path', default='base_rpm_weight')
    parser.add_argument('--save_iter', default=1, help=r'every iter to save model')
    cag = parser.parse_args()
    return cag


cag = config()


# def lr_decay(steps, scheduler):
#     mum_step = cag.top_epoch * (10553 / cag.train_bs + 1)
#     min_lr = cag.Min_LR
#     max_lr = cag.Max_LR
#     total_steps = cag.epoch * (10553 / cag.train_bs + 1)
#     if steps < mum_step:
#         lr = min_lr + abs(max_lr - min_lr) / (mum_step) * steps
#     else:
#         lr = scheduler.get_lr()
#         scheduler.step()
#     return lr
def lr_decay(steps):
    mum_step    = cag.top_epoch * (10553/cag.train_bs+1)
    min_lr      = cag.Min_LR
    max_lr      = cag.Max_LR
    total_steps = cag.epoch * (10553/cag.train_bs+1)
    if steps < mum_step:
        lr = min_lr + abs(max_lr - min_lr) / (mum_step) * steps
    else:
        lr = max_lr - abs(max_lr - min_lr) / (total_steps - mum_step + 1) * (steps - mum_step)
    return lr


def dice_loss(pred, mask):
    pred          = F.sigmoid(pred)
    intersection  = (pred * mask).sum(axis=(2, 3))
    unior         = (pred + mask).sum(axis=(2, 3))
    dice          = (2 * intersection + 1) / (unior + 1)
    dice          = paddle.mean(1 - dice)
    return dice


def boundary_dice_loss(pred, mask):
    pred = F.sigmoid(pred)
    n    = pred.shape[0]
    mask_boundary = paddle.nn.functional.max_pool2d(1 - mask, kernel_size=3, stride=1, padding=1)
    mask_boundary -= 1 - mask

    pred_boundary = paddle.nn.functional.max_pool2d(1 - pred, kernel_size=3, stride=1, padding=1)
    pred_boundary -= 1 - pred
    mask_boundary = paddle.nn.functional.max_pool2d(mask_boundary, kernel_size=5, stride=1, padding=2)
    pred_boundary = paddle.nn.functional.max_pool2d(pred_boundary, kernel_size=5, stride=1, padding=2)
    mask_boundary = paddle.reshape(mask_boundary, shape=(n, -1))
    pred_boundary = paddle.reshape(pred_boundary, shape=(n, -1))

    intersection  = (pred_boundary * mask_boundary).sum(axis=(1))
    unior         = (pred_boundary + mask_boundary).sum(axis=(1))
    dice          = (2 * intersection + 1) / (unior + 1)
    dice          = paddle.mean(1 - dice)
    return dice


def train(Dataset, Network, savepath):
    # dataset
    cag.min_mae = 10
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    cfg = Dataset.Config(
        snapshot=cag.mode_path, datapath=cag.train_dataset, savepath=savepath,
        mode='train', batch=cag.train_bs, lr=cag.Max_LR, momen=0.9, decay=5e-4, epoch=cag.epoch
    )

    data = Dataset.Data(cfg)
    loader = DataLoader(
        data,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=4,
       # collate_fn=data.collate
       use_shared_memory=True,
    )

    # network
    net = Network()
    net.train()

    # params
    total_params = sum(p.numel() for p in net.parameters())
    print('total params : ', total_params)

    # # optimizer
    # optimizer = paddle.optimizer.Momentum(parameters=net.parameters(), learning_rate=cag.Max_LR, momentum=cfg.momen,
    #                                       weight_decay=cfg.decay)
    # scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=cag.Max_LR,
    #                                                      T_max=len(loader) * (cag.epoch-cag.top_epoch), eta_min = cag.Min_LR)
    # global_step = 0
    # epoch_step = len(loader)
    # # training
    # with LogWriter(logdir=cag.log_dir) as writter:
    #     for epoch in range(0, cfg.epoch):
    #         start = datetime.datetime.now()
    #         for batch_idx, (image, mask) in enumerate(loader, start=1):
    #             lr = lr_decay(global_step, scheduler)
    #             optimizer.clear_grad()
    #             optimizer.set_lr(lr)

    #             writter.add_scalar(tag='train/lr', step=global_step, value=lr)
    #             global_step += 1
    #             out2 = net(image)
    #             loss = dice_loss(out2, mask) + boundary_dice_loss(out2, mask)
    #             loss.backward()
    #             optimizer.step()

    #             writter.add_scalar(tag='train/loss', step=global_step, value=loss.numpy()[0])

    #             if batch_idx % cag.show_step == 0:
    #                 msg = '%s | step:%d/%d/%d (%.2f%%) | lr=%.6f |  loss=%.6f  | %s ' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), batch_idx, epoch + 1, cfg.epoch, 
    #                 batch_idx / (10553 / cag.train_bs) * 100, optimizer.get_lr(), loss.item(),
    #                     image.shape)
    #                 print(msg)

    #         if epoch > cag.epoch / 3 * 2:
    #             paddle.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1) + '.pdparams')

    #         end = datetime.datetime.now()
    #         spend = int((end - start).seconds)
    #         mins = spend // 60
    #         secon = spend % 60
    #         print(f'this epoch spend {mins} m {secon} s. \n')
    # optimizer
    optimizer = paddle.optimizer.Momentum(parameters=net.parameters(), learning_rate=cag.Max_LR, momentum=cfg.momen,
                                          weight_decay=cfg.decay)
    global_step = 0
    # training
    all_losses = []
    all_lr = []
    all_metric = []
    for epoch in range(0, cfg.epoch):
        start = datetime.datetime.now()
        loss_list = []
        for batch_idx, (image, mask) in enumerate(loader, start=1):
            lr = lr_decay(global_step)
            optimizer.clear_grad()
            optimizer.set_lr(lr)
            all_lr.append(optimizer.get_lr())

            global_step += 1
            out2 = net(image)
            loss = dice_loss(out2, mask) +  boundary_dice_loss(out2, mask)#genggai
            loss.backward()
            loss_list.append(loss.numpy()[0])
            all_losses.append(loss.numpy()[0])
            optimizer.step()

            if batch_idx % cag.show_step == 0:
                msg = '%s | step:%d/%d/%d (%.2f%%) | lr=%.6f |  loss=%.6f | %s ' % (
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), batch_idx, epoch + 1, cfg.epoch,
                    batch_idx / (10553 / cag.train_bs) * 100, optimizer.get_lr(), loss.item(),
                     image.shape)
                print(msg)

        if epoch > cag.epoch / 3 * 2:
            paddle.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1) + '.pdparams')

        end = datetime.datetime.now()
        spend = int((end - start).seconds)
        mins = spend // 60
        secon = spend % 60
        loss_list = '%.5f' % np.mean(loss_list)
        print(f'this epoch spend {mins} m {secon} s and the average loss is {loss_list}', '\n')
    np.save('/home/aistudio/work/all_losses.npy', np.array(all_losses))
    np.save('/home/aistudio/work/all_lr.npy', np.array(all_lr))
    np.save('/home/aistudio/work/all_metric.npy', np.array(all_metric))

class Test(object):
    def __init__(self, Dataset, datapath, Network):
        self.datapath = datapath.split("/")[-1]
        self.datapath2 = datapath
        print(datapath)
        self.cfg = Dataset.Config(datapath=datapath, mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(
            self.data,
            batch_size=cag.test_bs,
            shuffle=True,
            num_workers=4,
            use_shared_memory=False)
        # network
        self.net = Network
        self.net.train()
        self.net.eval()

    def read_img(self, path):
        gt_img = self.norm_img(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        gt_img = (gt_img >= 0.5).astype(np.float32)
        return gt_img

    def norm_img(self, im):
        return cv2.normalize(im.astype('float'),
                             None,
                             0.0, 1.0,
                             cv2.NORM_MINMAX)

    def accuracy(self):
        with paddle.no_grad():
            mae = 0
            step = 0
            for image, mask, (H, W), maskpath in self.loader:

                out = self.net(image)
                pred = F.sigmoid(out[0])
                k_pred = pred
                for num in range(len(H)):
                    mae_pred = k_pred[num].unsqueeze(0)
                    mae_pred = F.interpolate(
                        mae_pred,
                        size=(
                            H[num].numpy()[0],
                            W[num].numpy()[0]),
                        mode='bilinear'
                        )
                    
                    path = self.datapath2 + '/mask/' + maskpath[num] + '.png'
                    mae_mask = paddle.to_tensor(self.read_img(path))
                    mae += (mae_pred[0][0] - mae_mask).abs().mean()
                    step += 1

            return mae.numpy()[0] / step


if __name__ == '__main__':
    #yuanshi
    # from ResBased import HRSODNet
    # train(dataset, HRSODNet, 'weight3')
    from SwinNet import HRSODNet
    train(dataset, HRSODNet, 'weightSwin3')

