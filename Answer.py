# -*- encoding: utf-8 -*-

# Author: Zhou XING
# E-mail: xingzhou-1998@outlook.com
# Last edition: Jan/18/2019

# Requirements:
# (1) Write down your answer below each question and make sure that the results
#     can be obtained by directly running this file.
# (2) It is better if you can establish a visual environment which consists of
#     an interpreter, and a set of installed packages.

# The following packages might be useful in this test:
import cv2
import os
import glob
import itertools
import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D,GlobalMaxPooling2D
from keras.layers import Activation, Dropout, Dense


main_path = './Zhou_XING_test_ml/'
os.chdir(main_path)
os.getcwd()
tex_color = (67, 51, 159)

# Question 1: The image q1 in the folder is a page which consists of five columns
# separated by four solid lines.

# (1) Detect the four lines in the image;

## (0) environment
Q_list = ['Q'+str(i) for i in range(1,4)]
for Qi in Q_list:
    path = os.path.join(main_path, 'To_Submit', Qi)
    if not os.path.exists(path):
        os.makedirs(path)
Q1_path = os.path.join(main_path, 'To_Submit', 'Q1')
Q2_path = os.path.join(main_path, 'To_Submit', 'Q2')
Q3_path = os.path.join(main_path, 'To_Submit', 'Q3')

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

# (2) Separate the page into 5 columns based on the four lines.

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
new_corner = points[-1][0] - (points[-3][0] - points[-4][0])  #Deal with the edge on img4
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



# Question 2: After finishing Question 1, you will get five images that are similar
# to the image q2 in the folder. Please use q2 to answer the following questions:

# (1) Compute the angle of the rotated text and rotate the image to correct
#     for the skew;
'''
THE angle will be showed when running the FOR loop.
'''

# (2) Calculate and plot row (y-axis) pixel dilation for the rotated image;
'''
PLEASE check the 'Dilation_imgi.png' file at Q2 folder.
'''

# (3) Further split the column image into rows (HINT: based on the pixel dilation
#     that you calculated, you can determine the position where you can split two
#     rows).
'''
PLEASE check the 'Q2>>Divided' folder.
'''

os.chdir(Q1_path)
files = ["img"+str(i)+".png" for i in range(5)]
for file in files:
    
    ## (0) environment
    
    new_path = os.path.join(Q2_path, 'Divided', file[:-4])
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ## (1) Angle
    
    ### Black_back_white_font
    th, threshed = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    pts = cv2.findNonZero(threshed)
    ret = cv2.minAreaRect(pts)

    (cx,cy), (w,h), ang = ret
    print(' ******************************************** \n')
    print('↓ The angle of the rotated text for '+ file +' ↓ \n')
    print(ang)
    print('\n ******************************************** ')

    ### White_back_black_font
    if file == 'img4.png': 
        thresh = 170
    else:
        thresh = 150
    im_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]   

    ## (2) Rotate
    
    if w>h:
        w,h = h,w
        ang += 90

    M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
    rotated = cv2.warpAffine(im_bw, M, (img.shape[1], img.shape[0]),borderValue=(255,255,255),borderMode=cv2.BORDER_CONSTANT)
  
    ## (3) Plot the Dilation
    
    hist = cv2.reduce(rotated,1,cv2.REDUCE_AVG).reshape(-1)
    x = np.array(range(len(hist)))
    fig, ax = plt.subplots(figsize=(15,10))
    ax.plot(x, hist,color='#9F393D')
    ax.set(xlabel = 'Rows', ylabel='Pixel Dilation',
           title = 'Row (y-axis) Pixel Dilation for ' + file)
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_ylim(255, 130)
    ax.grid()
    fig.savefig(os.path.join(Q2_path, "Dilation_" + file[:-4] + ".png"))
    
    print(' ******************************************** \n')
    print(' Dilation Plot for '+ file +' is saved !')
    print('\n ******************************************** ')
    
    
    ## (4) Division
    
    rotated_copy = rotated[:]
    th = 253
    H,W = img.shape[:2]
    uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
    lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]

    for y in uppers:
        cv2.line(rotated_copy, (0,y), (W, y), (0,0,0), thickness=3)
    
    for y in lowers:
        cv2.line(rotated_copy, (0,y), (W, y), (0,0,0), thickness=1)
    
    cv2.imwrite(os.path.join(Q2_path,file[:-4]+'_lined_.png'),rotated_copy)

    for i in range(len(uppers)):   
        if i+1 == len (uppers):
            break
        new_img = rotated[lowers[i]:uppers[i-len(lowers) + len(uppers)+1],:]
        cv2.imwrite(os.path.join(new_path,file[:-4] + "_div_" + str(i) + '.png'), new_img)
    
    print(' ******************************************** \n')
    print(' Row Division for '+ file +' is done !')
    print('\n ******************************************** ')
    


# Question 3: Based on the format of characters, we can divided row images into seven
# categories: 0. blanks, 1. page number, 2. company name, 3. addresses, 4. variable
# names, 5. variable value, 6. personnel (image q3 in the folder provides a more
# detailed explanation for you). More examples for each category are provided in the
# folder named as "trainset".
# For the rows that you get from Question 2, write a code so that all the rows will be
# automatically classified into the seven categories (HINT: package keras could work
# for this)

## (1) png to jpg & categorization
    
path_ori = os.path.join(main_path, 'trainset', 'trainset')
upper_path = os.path.join(main_path, 'trainset')
jpg_folder = os.path.join(upper_path,'jpg')

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10
np.random.seed(0)

for name in os.listdir(path_ori):
    path = path_ori + name
    for classif in ['training', 'validation', 'testing']:   
        if not os.path.exists(os.path.join(jpg_folder,classif,name)):
            os.makedirs(os.path.join(jpg_folder, classif, name))
    for img in os.listdir(path):
        im = cv2.imread(path + '/' + img)
        chance = np.random.randint(100)
        if chance < VALIDATION_PERCENTAGE:
            os.chdir(os.path.join(jpg_folder, 'validation', name))
            cv2.imwrite(img[:-3]+'jpg',im)
        elif chance < (TEST_PERCENTAGE + VALIDATION_PERCENTAGE):
            os.chdir(os.path.join(jpg_folder, 'testing', name))
            cv2.imwrite(img[:-3]+'jpg',im)
        else:
            os.chdir(os.path.join(jpg_folder, 'training', name))
            cv2.imwrite(img[:-3]+'jpg',im)

print('\n****************************\n')
print(' All png transverted to jpg !')
print('\n****************************')

## (2) Model training

training_path = os.path.join(jpg_folder, 'training')
validation_path = os.path.join(jpg_folder, 'validation')
os.chdir(Q3_path)

batch_size = 20
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
        training_path,  
        target_size = (170, 350),  # all images will be resized to 170x350
        batch_size = batch_size, class_mode = 'categorical'
        )  

validation_generator = test_datagen.flow_from_directory(
        validation_path,
        target_size = (170, 350),
        batch_size = batch_size,class_mode = 'categorical'
        )

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (170, 350, 3), data_format = 'channels_last'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(GlobalMaxPooling2D())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(7))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch = 800 // batch_size,
        epochs = 30,
        validation_data = validation_generator,
        validation_steps = 800 // batch_size)
model.save('model.h5')

class_dict = train_generator.class_indices
print(class_dict)
inv_dic = {v: k for k, v in class_dict.items()}
print(inv_dic)

## (3) Prediction (classification)

divided_path=os.path.join(main_path, 'To_Submit','Q2','Divided')

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(170, 350))
    img_tensor = image.img_to_array(img)                    
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

subdirs = [x[0] for x in os.walk(divided_path)][1:]
oneinput = [os.path.join(pathy, '*.png') for pathy in subdirs]
input_1 = []
input_1.extend(glob.glob(oneinput[i]) for i in range(len(oneinput)))
input_real = []
for i in input_1:
    for j in i:
        input_real.append(j)
        
img_tensors = []
for img_path in input_real:
    img_tensors.append(load_image(img_path)[0])
img_tensor = np.asarray(img_tensors)
print(img_tensors.shape)

print(' ******************************************** \n')
print('         All divided pics are ready !')
print('\n ******************************************** ')

model = load_model('model.h5')
classes = model.predict_classes(img_tensors, batch_size=20)
print(classes)
print(inv_dic)
labeled_class = [inv_dic[classes[i]] for i in range(len(classes))]
print(labeled_class) 

path_to_save = os.path.join(Q3_path, 'ML_classed')

if len(labeled_class)==len(input_real):
    print('\n**************\n'+'   Success!'+'\n**************\n')
else:
    print('\n**************\n'+'   Problems!'+'\n**************\n')
    
for i in range(len(labeled_class)):
     if not os.path.exists(os.path.join(path_to_save, labeled_class[i])):
            os.makedirs(os.path.join(path_to_save, labeled_class[i]))
     basename = os.path.basename(input_real[i])
     img = cv2.imread(input_real[i])
     cv2.imwrite(os.path.join(path_to_save, labeled_class[i],basename), img)

print(' ******************************************** \n')
print('       All divided pics are classified !')
print('\n ******************************************** ')


''' 
PATH TO CHECK

    main_path = './Zhou_XING__test_ml/'
    Q1_path = os.path.join(main_path, 'To_Submit', 'Q1')
    Q2_path = os.path.join(main_path, 'To_Submit', 'Q2')
    Q3_path = os.path.join(main_path, 'To_Submit', 'Q3')
    path_ori = os.path.join(main_path, ’trainset', 'trainset')
    upper_path = os.path.join(main_path, ’trainset')
    jpg_folder = os.path.join(upper_path,'jpg')
    divided_path = os.path.join(main_path, 'To_Submit', 'Q2', 'Divided')
    training_path = os.path.join(jpg_folder, 'training')
    validation_path = os.path.join(jpg_folder, 'validation')

'''