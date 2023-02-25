import cv2

pic = cv2.imread('../oxford5k/oxbuild_images-v1/all_souls_000000.jpg')
sift = cv2.xfeatures2d.SIFT_create(nfeatures=500, contrastThreshold=0.01, edgeThreshold=30, sigma=1.6)
kp, des = sift.detectAndCompute(pic, None)
print(kp)
print(des)