import os
import cv2
import numpy as np

main_path = 'F:/Travail/Harvard_RA/test_ml/'
path_ori = main_path + '/trainset/trainset/'
upper_path = main_path + 'trainset/'
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
   
#
#       training_images = []
#        testing_images = []
#        validation_images = []
#        for file_name in file_list:
#            base_name = os.path.basename(file_name) # it is a str, name of the file
#            chance = np.random.randint(100)
#            if chance < validation_percentage:
#                validation_images.append(base_name)
#            elif chance < (testing_percentage + validation_percentage):
#                testing_images.append(base_name)
#            else:
#                training_images.append(base_name)