import skimage.io
from skimage.feature import (match_descriptors, ORB, SIFT, plot_matches)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

#Visualizing Matches
def vis_ftr_matches(img1_rgb, img2_rgb, keypoints1, keypoints2, match_img12):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plot_matches(ax, img1_rgb, img2_rgb, keypoints1, keypoints2, match_img12)
    plt.show()

img1 = "data/uwdata/20150419T031538.000Z.jpg"
img2 = "data/uwdata/20150419T031608.000Z.jpg"
def feature_matching(img1, img2, descriptor):
    # Import Image Pairs
    img1_rgb = skimage.io.imread(img1)
    img2_rgb = skimage.io.imread(img2)
    # Convert RGB to gray
    img1_gray = rgb2gray(img1_rgb)
    img2_gray = rgb2gray(img2_rgb)

    if descriptor == "ORB":
        descriptor_ext = ORB(n_keypoints=100)
    if descriptor == "SIFT":
        descriptor_ext = SIFT()
    else:
        raise ValueError('Invalid descriptor type, only "SIFT", "ORB" valid')
    #For image one
    descriptor_ext.detect_and_extract(img1_gray)
    keypoints1 = descriptor_ext.keypoints
    descriptor1 = descriptor_ext.descriptors
    #for image two
    descriptor_ext.detect_and_extract(img2_gray)
    keypoints2 = descriptor_ext.keypoints
    descriptor2 = descriptor_ext.descriptors
    #Keypoint Matching
    if descriptor == "ORB":
        match_img12 = match_descriptors(descriptor1, descriptor2, cross_check=True)
    if descriptor == "SIFT":
        match_img12 = match_descriptors(descriptor1, descriptor2, max_ratio=0.5, cross_check=True)

    vis_ftr_matches(img1_rgb, img2_rgb, keypoints1, keypoints2, match_img12)
    print(match_img12.shape)
feature_matching(img1, img2, "SIFT")
feature_matching(img1, img2, "ORB")