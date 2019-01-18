# load_model_sample.py
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import cv2

main_path = 'F:/Travail/Harvard_RA/test_ml/'
Q3_path = os.path.join(main_path, 'To_Submit','Q3')
divided_path=os.path.join(main_path, 'To_Submit','Q2','Divided')
os.chdir(Q3_path)


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