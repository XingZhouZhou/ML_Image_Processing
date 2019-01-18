import os 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,GlobalMaxPooling2D
from keras.layers import Activation, Dropout, Dense

main_path = 'F:/Travail/Harvard_RA/test_ml/'
Q3_path = os.path.join(main_path, 'To_Submit','Q3')
upper_path = main_path + 'trainset/'
jpg_folder = os.path.join(upper_path, 'jpg')
training_path = os.path.join(jpg_folder, 'training')
validation_path = os.path.join(jpg_folder, 'validation')
img_path = 'F:/Travail/Harvard_RA/test_ml/trainset/jpg'
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
