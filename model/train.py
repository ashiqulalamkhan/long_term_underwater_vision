import torch
import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from unet import UNet
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torchsummary import summary
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import sift_pose_est
import kornia
import k_sift
global loss_global
loss_global = torch.tensor([0])

#torch.set_default_dtype(torch.float16)
#torch.set_float32_matmul_precision('medium')

#For Positive and Negative data separate
file1 = open('/home/turin/Documents/GitHub/long_term_underwater_vision/dataset/dataset_curated', 'rb')
pos_data = pickle.load(file1)
file1.close
im_pos, label_pos = pos_data
im_pos_reduced = im_pos[:10]
label_pos_reduced = label_pos[:10]
transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float16), transforms.Resize(size=(1360,1024), interpolation=transforms.InterpolationMode("nearest"))])


###DATASET AND DATALOADER CREATION
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data_path = "/home/turin/Desktop/lizard_island/jackson/chronological/2014/r20141102_074952_lizard_d2_081_horseshoe_circle01/081_photos/"
        img1_path = data_path+self.data[idx][0]
        img2_path = data_path+self.data[idx][1]
        im_bgr1 = cv2.imread(img1_path)
        im_bgr2 = cv2.imread(img2_path)
        im_gray1 = cv2.cvtColor(im_bgr1, cv2.COLOR_BGR2GRAY)
        im_gray2 = cv2.cvtColor(im_bgr2, cv2.COLOR_BGR2GRAY)
        img1 = transform(im_gray1)
        img2 = transform(im_gray2)
        label = self.labels[idx][0]
        pose_img1 = self.labels[idx][1][0]
        pose_img2 = self.labels[idx][1][1]
        #print(img1, img2, label, pose_img1, pose_img2)
        return (img1, img2),(label, pose_img1, pose_img2, (img1_path, img2_path))

training_data = CustomDataset(im_pos_reduced,label_pos_reduced)
train_dataloader = DataLoader(training_data, batch_size=1, shuffle=False, num_workers=0)

###MODEL CREATION
def create_model():
    # model = torchvision.models.resnet18(pretrained=False, num_classes=2)
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model.maxpool = nn.Identity()
    model2 = UNet(in_channels=1,
             out_channels=1,
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2)
    return model2
model = create_model()
model.cuda()
####SAVING FROM CHECKPOINT
#summary(m,(3,256,384))
checkpoint = torch.load("/home/turin/Documents/GitHub/long_term_underwater_vision/model/auto_logs/autoencoder/version_9/checkpoints/epoch=2-step=2400.ckpt")
check_state = checkpoint['state_dict']
model_state = model.state_dict()
key_list = list(check_state.keys())
len(key_list)
for idx in key_list:
    # print(idx)
    # print(idx[6:
    model_state[idx[6:]] = check_state[idx]
model.load_state_dict(model_state)


###CUSTOM LOSS FUNCTION

class CustomPoseLoss(Function):
    @staticmethod
    def forward(ctx, pred0, pred1, pose):
        # rt_loss = 0
        # for idx in range(len(input1)):
        #     im1 = input1[idx].data
        #     im2 = input2[idx].data
        #     im1 = torch.permute(im1, (1, 2, 0))
        #     im2 = torch.permute(im2, (1, 2, 0))
        #     im1 = im1.cpu().numpy() * 255
        #     im2 = im2.cpu().numpy() * 255
        #     im1 = im1.astype(np.uint8)
        #     im2 = im2.astype(np.uint8)
        #     mat1 = pose[0][idx].numpy(force=True)
        #     mat2 = pose[1][idx].numpy(force=True)
        #     x, y, l_pts = sift_pose_est.get_relative_angle(im1, im2, mat1.squeeze(), mat2.squeeze(), display=True)
        #     if l_pts > 15:
        #         #rt_loss += .1 * (3 * x + 7 * np.sum(y) / 9)
        #         rt_loss += np.sum(y) / 9
        #     else:
        #         rt_loss += 30 / (l_pts + 1)
        pred0_np = pred0.detach().cpu().numpy()
        pred1_np = pred1.detach().cpu().numpy()
        #pose0_np = pose[0].detach().cpu().numpy()
        #pose1_np = pose[1].detach().cpu().numpy()
        #rest_loss, test_loss, rabs_loss, tabs_loss = cal_pose(pred0_np, pred1_np, pose0_np, pose1_np)
        #rest_loss = torch.tensor(rest_loss, requires_grad=True).cuda()
        #rabs_loss = torch.tensor(rabs_loss, requires_grad=False).cuda()
        ctx.save_for_backward(pred0, pred1)
        result = np.mean(pred0_np-pred1_np)
        return pred0.new(result)
    @staticmethod
    def backward(ctx, grad_output):
        numpy_go = grad_output.cpu().numpy()
        #rest_loss, rabs_loss,
        pred0, pred1 = ctx.saved_tensors
        grad_pred0 = grad_pred1 = None
        #grad_pred0 = (rabs_loss-rest_loss)* grad_output*torch.ones(pred0.shape, requires_grad=False).cuda()
        #grad_pred1 = (rabs_loss-rest_loss)* grad_output*torch.ones(pred1.shape, requires_grad=False).cuda()
        #grad_pred0 = grad_output*pred0
        #grad_pred1 = grad_output*pred1
        print(grad_pred0)
        return grad_output, grad_output, None#, None, None#grad_output.new(output), grad_input1, grad_input2, grad_pose

def custom_loss(pred0, pred1, k1, k2, pose1, pose2):#, pose):
    #pred0_np = pred0.data.cpu().numpy()
    #pred1_np = pred1.data.cpu().numpy()
    # result = np.subtract(pred0_np,pred1_np)
    # #result = pred0.new(torch.tensor([result]).cuda())
    # #result.requires_grad = True
    # loss = torch.from_numpy(np.asarray(np.abs(np.sum(result))))
    # loss.requires_grad =  True
    pr1 = pred0.type(torch.float32).cpu()
    pr2 = pred1.type(torch.float32).cpu()
    #loss_ncc = torch.sum((pr1-pr1.mean())*(pr2-pr2.mean()))/(1024*1360*torch.std(pr1)*torch.std(pr2))
    #loss1 = F.mse_loss(loss_ncc, torch.tensor(1.0))
    plt.imshow(pr1.squeeze(0).squeeze(0).data, cmap="gray")
    plt.show()
    # plt.imshow(pr2.squeeze(0).squeeze(0).data, cmap="gray")
    # plt.show()
    R, t, pts = k_sift.im_resize(pr1, pr2, k1, k2)
    T_ground = torch.linalg.solve(pose2, pose1)
    R_ground = T_ground[:,:3,:3]
    t_ground = T_ground[:,:3,3]/torch.linalg.norm(T_ground[:,:3,3])
    if R==None:
        loss2 = 2.4/torch.sum((pts+0.0001)/(pts+0.0001))
    else:
        loss2 = torch.mean(torch.abs(torch.sub(R, R_ground.cpu()))) + .01*torch.mean(torch.abs(torch.sub(t, t_ground.cpu())))
    # # #print(l.shape)
    #if l.any():
        #loss = torch.mean(torch.sub(pred0, pred1))
        #loss = torch.mean(torch.sub(l, t))
        #return loss #pred0.new(torch.tensor([result]).cuda())
        #return CustomPoseLoss.apply(pred0, pred1, pose)
    #else:
        #return torch.sub(100,20)
    return loss2#loss1+10*loss2


def cal_pose(input1, input2, pose1, pose2):
    rest_loss = []
    test_loss = []
    rabs_loss = []
    tabs_loss = []
    for idx in range(len(input1)):
        # im1 = input1[idx].data
        # im2 = input2[idx].data
        im1 = input1[idx].transpose(1, 2, 0)
        im2 = input2[idx].transpose(1, 2, 0)
        im1 = im1 * 255
        im2 = im2 * 255
        im1 = im1.astype(np.uint8)
        im2 = im2.astype(np.uint8)
        mat1 = pose1[idx]
        mat2 = pose2[idx]
        # x, y, l_pts = sift_pose_est.get_relative_angle(im1, im2, mat1.squeeze(), mat2.squeeze(), display=False)
        # if l_pts > 15:
        #     #rt_loss += .1 * (3 * x + 7 * np.sum(y) / 9)
        #     rt_loss += np.sum(y) / 9
        # else:
        #     rt_loss += 30 / (l_pts + 1)
        R_est, t_est, R_abs, t_abs = sift_pose_est.get_relative_angle(im1, im2, mat1.squeeze(), mat2.squeeze(), display=False)
        rest_loss.append(R_est)
        rabs_loss.append(R_abs)
        #r_loss+=F.mse_loss(torch.from_numpy(R_est), torch.from_numpy(R_abs))
    return rest_loss, test_loss, rabs_loss, tabs_loss


###LIGHTNING MODEL
class LitResnet(pl.LightningModule):
    def __init__(self, lr=0.95):
        super().__init__()
        self.example_input_array = torch.rand((2, 1, 1, 1024, 1360), dtype = torch.float16)##For vis tensor board computation graph
        self.hparams_lr = lr
        self.save_hyperparameters()
        self.model = model
        self.K = torch.tensor([[1.7181e+03, 0.0000e+00, 6.8339e+02],
        [0.0000e+00, 1.7173e+03, 5.4039e+02],
        [0.0000e+00, 0.0000e+00, 1.0000e+00]])
    # def poseloss(self,im, pred, pose):
    #     rt_loss = 0
    #     for idx in range (len(pred[0])):
    #         im1 = torch.permute(pred[0][idx],(1,2,0))
    #         im2 = torch.permute(pred[1][idx],(1,2,0))
    #         im1 = im1.numpy(force=True)*255
    #         im2 = im2.numpy(force=True)*255
    #         im1 = im1.astype(np.uint8)
    #         im2 = im2.astype(np.uint8)
    #         mat1 = pose[0][idx].numpy(force=True)
    #         mat2 = pose[1][idx].numpy(force=True)
    #         x ,y, l_pts  = sift_pose_est.get_relative_angle(im1, im2, mat1.squeeze(), mat2.squeeze(),display=True)
    #         if l_pts>15:
    #             rt_loss += .1*(3*x + 7*np.sum(y)/9)
    #         else:
    #             rt_loss += 30/(l_pts+1)
    #     #print(rt_loss)
    #     return rt_loss #F.mse_loss(pred[0]-im[0], pred[1]-im[1]) #rt_loss/len(pred[0])

    def forward(self, x):
        #print(x[0].shape, x[1].shape)
        out1 = self.model(x[0])
        out2 = self.model(x[1])
        return out1, out2
    def training_step(self, batch, batch_idx):
        x, y = batch
        pose = (y[1], y[2])
        pred= self.forward(x)
        #c_loss = cal_pose(pred[0], pred[1], pose)
        #print(c_loss[2])
        prd = custom_loss(pred[0], pred[1], self.K, self.K, pose[0], pose[1])#, pose)
        #loss1 = F.mse_loss(prd[0], torch.zeros(prd[0].shape).cuda())
        #loss2 = F.mse_loss(prd[1], torch.zeros(prd[1].shape).cuda())
        loss = prd # + loss2#torch.tensor(c_loss[2]).cuda()) #+ F.mse_loss(pred[1], est1)
        self.log("my_loss", loss, prog_bar=True)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams_lr,eps=1e-04, weight_decay=0.0005)
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     lr=self.hparams_lr,
        #     momentum=0.9,
        #     weight_decay=5e-4,
        # )
        # steps_per_epoch = 10
        # scheduler_dict = {
        #     "scheduler": OneCycleLR(
        #         optimizer,
        #         0.1,
        #         epochs=self.trainer.max_epochs,
        #         steps_per_epoch=steps_per_epoch,
        #     ),
        #     "interval": "step",
        # }
        return {"optimizer": optimizer}#, "lr_scheduler": scheduler_dict}


###TRAINING
model = LitResnet(lr=0.0005)
model = model.type(torch.float16)
from pytorch_lightning.loggers import TensorBoardLogger
logger = TensorBoardLogger("tb_logs",log_graph=True, name="my_model")
trainer = Trainer(
    max_epochs=100,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    logger=logger,
    callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
    log_every_n_steps= 1
)
trainer.fit(model, train_dataloader)

from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')