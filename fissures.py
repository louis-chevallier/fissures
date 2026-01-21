from utillc import *
import cv2, os
import numpy as np

folder = "/home/louis/Desktop/tmp/fissures"
imaf = "IMG_20251107_113638.jpg"
imbf = "IMG_20250328_170711.jpg"
imf = [ imaf, imbf]

ims = [ cv2.imread(os.path.join(folder, i)) for i in imf]
image = ims[0]
EKOI(image)
EKOX(image.shape)

# Source - https://stackoverflow.com/q/76568361
# Posted by Daniel Agam, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-21, License - CC BY-SA 4.0


EKOX(ims[0].shape)

# Convert images to grayscale
img1, img2 = [ cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in ims]
EKOX(img1.shape)

# crop the images to ROI - The ROI is the same for both images and is the left bottom corner in size of 1/5 of the image
img1_roi = img1[0:int(img1.shape[0] / 5), 0:int(img1.shape[1] / 5)]
img2_roi = img2[0:int(img2.shape[0] / 5), 0:int(img2.shape[1] / 5)]

img1_roi, img2_roi = [ e[0:int(e.shape[0] / 5), 0:int(e.shape[1] / 5)] for e in [ img1_roi, img2_roi]]


# Calculate the mean of the images.
mean_img1 = np.mean(img1_roi)
mean_img2 = np.mean(img2_roi)

mean_img1, mean_img2 = [ np.mean(e) for e in (img1_roi, img2_roi) ]

# Calculate the ratio of the brightness of the images.
ratio = mean_img1 / mean_img2
print(f'Brightness ratio: {ratio}')

# Multiply the second image by the ratio.
img2 = img2 * ratio

# Convert the image to uint8 again.
img2 = np.clip(img2, 0, 255)
img2 = img2.astype(np.uint8)


EKOX(img1.shape)


# Source - https://stackoverflow.com/q/76568361
# Posted by Daniel Agam, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-21, License - CC BY-SA 4.0

image1 = img1 #cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
image2 = img2 #cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
height, width = image2.shape

sift = cv2.SIFT_create()
EKO()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
EKO()
bf = cv2.BFMatcher()

matches = bf.knnMatch(descriptors1, descriptors2, k=2)
EKOX(len(matches))
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
EKOX(M.shape)

draw_params = dict(singlePointColor=None, flags=2)
img3 = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, **draw_params)
EKOI(img3)

# Source - https://stackoverflow.com/q/76568361
# Posted by Daniel Agam, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-21, License - CC BY-SA 4.0



aligned_image = cv2.warpAffine(image2, M[0:2, :], (image1.shape[1], image1.shape[0]))
aligned_image2 = cv2.warpPerspective(image2, M, (image1.shape[1], image1.shape[0]))

EKOI(aligned_image)
EKOI(aligned_image2)
