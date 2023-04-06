import torch
import kornia
import cv2
import numpy as np
import pickle
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import kornia.feature as K
import torchvision
from kornia.feature import *
from torchvision import transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float32), transforms.Resize(size=(640,512), interpolation=transforms.InterpolationMode("nearest"))])


device = torch.device('cpu')
datatype = torch.float32
##MATCH VISUALIZATION
def vis_k_sift(img1, img2, match):
    img_matched = 0
    img_stacked = 0
    kp1 = match[0].data.cpu().numpy().astype(int)
    kp2 = match[1].data.cpu().numpy().astype(int)
    pts1 = kp1
    pts2 = kp2
    pts2[:,0] = kp2[:, 0]+512
    img_stacked = np.hstack((img1, img2))
    for x in range(len(kp1)):
        img_matched = cv2.arrowedLine(img_stacked, pts1[x], pts2[x], (0, 0, 255), 3)
    plt.imshow(img_matched)
    plt.show()


#%%
def get_camera_mat(file):
    tree = ET.parse(file)
    root = tree.getroot()
    for x in  root.iter("Camera_Matrix"):
        for y in x.iter("data"):
            camera_mat = np.matrix(y.text)
    for x in  root.iter("Distortion_Coefficients"):
        for y in x.iter("data"):
            distort_mat = np.matrix(y.text)
    camera_mat = np.asarray(camera_mat.reshape((3,3)))
    distort_mat = np.asarray(distort_mat)
    return torch.tensor(camera_mat, dtype=datatype, device=device), torch.tensor(distort_mat,dtype=datatype, device=device)

def tensor_sift(img1, img2):
    img1 = kornia.image_to_tensor(img1).float().unsqueeze(0).to(device)
    img2 = kornia.image_to_tensor(img2).float().unsqueeze(0).to(device)
    #f img1.shape()
    sift = kornia.feature.SIFTDescriptor(32, rootsift=True).to(device)
    descriptor = sift

    resp = BlobHessian()
    scale_pyr = kornia.geometry.ScalePyramid(3, 1.5, 512, double_image=True)
    nms = kornia.geometry.ConvQuadInterp3d(10)
    n_features = 600
    detector = ScaleSpaceDetector(n_features,
                                resp_module=resp,
                                nms_module=nms,
                                scale_pyr_module=scale_pyr,
                                ori_module=kornia.feature.LAFOrienter(19),
                                mr_size=6.0).to(device)

    with torch.no_grad():
        lafs, resps = detector(torch.cat([img1, img2]))
        patches =  kornia.feature.extract_patches_from_pyramid(torch.cat([img1, img2]), lafs, 32)
        B, N, CH, H, W = patches.size()
        # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
        # So we need to reshape a bit :)
        descs = descriptor(patches.view(B * N, CH, H, W)).view(B, N, -1)
        scores, matches = kornia.feature.match_snn(descs[0], descs[1], 0.75)#matches = Indexes of [desc1, desc2]

    # Now RANSAC
    src_pts = lafs[0,matches[:,0], :, 2]#.data.cpu().numpy()
    dst_pts = lafs[1,matches[:,1], :, 2]#.data.cpu().numpy()
    #F = kornia.geometry.epipolar.find_fundamental(src_pts, dst_pts)
    print(src_pts.shape)
    #H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0, 0.999, 10000)
    #inliers = matches[torch.from_numpy(mask).bool().squeeze(), :]
    return src_pts, dst_pts

def im_resize(im1, im2, K1, K2):
    # print(im1.shape)
    # img1 = torchvision.transforms.functional.resize(im1, [512, 680])
    # img2 = torchvision.transforms.functional.resize(im2, [512, 680])
    # print(im1.shape)
    sift = kornia.feature.SIFTDescriptor(32, rootsift=True).to(device)
    descriptor = sift
    resp = BlobHessian()
    scale_pyr = kornia.geometry.ScalePyramid(3, 1.5, 512, double_image=True)
    nms = kornia.geometry.ConvQuadInterp3d(10)
    n_features = 600
    detector = ScaleSpaceDetector(n_features,
                                  resp_module=resp,
                                  nms_module=nms,
                                  scale_pyr_module=scale_pyr,
                                  ori_module=kornia.feature.LAFOrienter(19),
                                  mr_size=6.0).to(device)

    # with torch.no_grad():
    lafs, resps = detector(torch.cat([im1, im2]))
    patches = kornia.feature.extract_patches_from_pyramid(torch.cat([im1, im2]), lafs, 32)
    B, N, CH, H, W = patches.size()
    # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
    # So we need to reshape a bit :)
    descs = descriptor(patches.view(B * N, CH, H, W)).view(B, N, -1)
    scores, matches = kornia.feature.match_snn(descs[0], descs[1], 0.75)  # matches = Indexes of [desc1, desc2]

    # Now RANSAC
    src_pts = lafs[0, matches[:, 0], :, 2]  # .data.cpu().numpy()
    dst_pts = lafs[1, matches[:, 1], :, 2]  # .data.cpu().numpy()
    if len(src_pts)>12:
        F = kornia.geometry.epipolar.find_fundamental(src_pts.unsqueeze(0), dst_pts.unsqueeze(0))
        E = kornia.geometry.epipolar.essential_from_fundamental(F, K1.unsqueeze(0), K2.unsqueeze(0))
        R, t, _ =  kornia.geometry.epipolar.motion_from_essential_choose_solution(E, K1.unsqueeze(0), K2.unsqueeze(0), src_pts.unsqueeze(0), dst_pts.unsqueeze(0))
        return R, t, src_pts
    else:
    # H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0, 0.999, 10000)
    # inliers = matches[torch.from_numpy(mask).bool().squeeze(), :]
        return None, None, src_pts


def main():
    calib_file = "/home/turin/Desktop/lizard_dataset_curated/opencv_cam_calib.xml"
    cam_K, cam_D = get_camera_mat(calib_file)

    img_path = "/home/turin/Desktop/lizard_island/jackson/chronological/2014/r20141102_074952_lizard_d2_081_horseshoe_circle01/081_photos/"
    file = open("/home/turin/Documents/GitHub/long_term_underwater_vision/dataset/dataset_positive", 'rb')
    im, label = pickle.load(file)
    file.close()
    idx = 4095
    img1 = img_path + im[idx][0]
    img2 = img_path + im[idx][1]
    print(img1, img2)
    global mat1
    mat1 = label[idx][1][0]
    global mat2
    mat2 = label[idx][1][1]
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    print(gray1.shape)
    gray1_reduced = cv2.resize(gray1, (680, 512))
    gray2_reduced = cv2.resize(gray2, (680, 512))
    clip_im = int(.5 * (gray2_reduced.shape[1] - gray2_reduced.shape[0]))
    gray1_clipped = gray1_reduced[:, clip_im:gray1_reduced.shape[0] + clip_im]
    gray2_clipped = gray2_reduced[:, clip_im:gray1_reduced.shape[0] + clip_im]
    ##FOR TORCH
    gray1_t = transform(gray1_reduced)
    gray2_t = transform(gray2_reduced)
    print(gray1_t.shape)
    matches = im_resize(gray1_t.unsqueeze(0), gray2_t.unsqueeze(0), cam_K, cam_K)
    print(matches[0].shape)
    #vis_k_sift(gray1_clipped, gray2_clipped, matches)
    return 0#(matches[0].shape)



if __name__ == "__main__":
    main()