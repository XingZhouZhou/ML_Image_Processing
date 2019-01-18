import cv2
import os
import itertools
import numpy as np
from matplotlib import pyplot as plt

main_path = 'F:/Travail/Harvard_RA/test_ml/'
os.chdir(main_path)
os.getcwd()
tex_color = (67, 51, 159)

# (0) environment
Q_list = ['Q'+str(i) for i in range(1,5)]
for Qi in Q_list:
    path = os.path.join(main_path, 'To_Submit', Qi)
    if not os.path.exists(path):
        os.makedirs(path)
Q1_path = os.path.join(main_path, 'To_Submit', 'Q1')

## (1) read
img = cv2.imread("q1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#plt.figure(figsize=(17,17))
#plt.imshow(gray,cmap='gray')
os.chdir(Q1_path)

## (2) threshold
th, threshed = cv2.threshold(gray, 200, 20, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
#th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,5)
#plt.figure(figsize=(12,12))
#plt.imshow(threshed,cmap='gray')

## (3) division lines
lines = cv2.HoughLines(threshed, rho=1, theta=np.pi/700, threshold = 2000)
amplifier = img.size/400
keys=[]
img_copy = img.copy()
for i in range(len(lines)):
    for rho, theta in lines[i]:
        if theta<0.05:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + amplifier*(-b))
            y1 = int(y0 + amplifier*(a))
            x2 = int(x0 - amplifier*(-b))
            y2 = int(y0 - amplifier*(a))
            k0 = (x2-x1)/(y2-y1)
            keys.append((k0, y1, x1))
            cv2.line(img_copy, (x1, y1), (x2, y2), tex_color, 4)

            
cv2.imwrite("segmented.png", img_copy)  
plt.figure(figsize=(24.17, 17))
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
plt.subplot(111), plt.imshow(img_copy)
plt.title('hough'), plt.xticks([]), plt.yticks([])

## (4) split image
def gen_points(i):
    k0, y1, x1 = keys[i][:]
    x_1, x_2 = [(j-y1)*k0+x1 for j in [0, img.shape[0]]]
    return [[int(x_1), 0], [int(x_2), img.shape[0]]]
points = []
for i in range(len(keys)):
    points += gen_points(i)

corner_0 = list(itertools.product([0, img.shape[1]], [0, img.shape[0]]))
corner = [list(i) for i in corner_0]
points += corner
points = sorted(points, key = lambda x: (x[0],-x[1]))

bias=200
####
points[-1] = [img.shape[1] - bias, 0]
new_corner = points[-1][0] - (points[-3][0] - points[-4][0])
points[-2] = [new_corner, points[-4][1]]
####

print(points)


def division(keys):
    for i in range(len(keys)+1):
        mask = np.zeros(img.shape[:2], np.uint8)
        pst = []
        for j in [2*i+1, 2*i, 2*i+2, 2*i+3]:
            pst.append(points[j]) 
        pst = np.array(pst)
        #print('********************\n\n')
        #print(pst)
        #print('\n\n********************')
        x = [pst[i][0] for i in range(len(pst))]
        _ = cv2.drawContours(mask, np.int32([pst]), 0, 255, -1)
        img1 = img.copy()
        img1[mask == 0] = 255 #turn the rest to white
        crop_img = img1[:, min(x):max(x)]
        cv2.imwrite("img"+str(i) + ".png", crop_img)
    print('\n********************\n\n')
    print(' Division is done !')
    print('\n\n********************')

division(keys)

img4 = cv2.imread("img4.png")
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
plt.figure(figsize = (2.5, 16))
plt.imshow(img4)
plt.title('Sample: img4'), plt.xticks([]), plt.yticks([])


print('\n********************\n\n')
print(' Please move to Q2 !')
print('\n\n********************')


# =============================================================================
# 
# copy = cv2.imread("q1.png")
# cv2.line(copy,points[6],points[7],(255,0,0),thickness=10)
# plt.figure(figsize=(13, 13))
# copy = cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)
# plt.subplot(111),plt.imshow(copy)
# =============================================================================
