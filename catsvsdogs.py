import os
import shutil  # 复制文件
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,BatchNormalization, Activation
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing import image

"""获取数据"""
# 原始目录所在的路径 数据集未压缩
original_dataset_dir_cat = 'kaggle_original_data_cat'
original_dataset_dir_dog = 'kaggle_original_data_dog'
 
# 存储较小数据集的目录
base_dir = 'data'
os.mkdir(base_dir)
 
# 训练、验证、测试数据集的目录
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)
 
# 猫训练图片所在目录
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
# 狗训练图片所在目录
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)
 
# 猫验证图片所在目录
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
# 狗验证数据集所在目录
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)
 
# 猫测试数据集所在目录
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
# 狗测试数据集所在目录
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)
 

# 复制最开始的3000张猫图片到 train_cats_dir
fnames = ['{}.jpg'.format(i) for i in range(3000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_cat, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)
 
# 复制接下来1000张猫图片到 validation_cats_dir
fnames = ['{}.jpg'.format(i) for i in range(3000, 4000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_cat, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
 
# 复制接下来1000张图片到 test_cats_dir
fnames = ['{}.jpg'.format(i) for i in range(4000, 5000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_cat, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
 

# 复制最开始的5000张狗图片到 train_dogs_dir
fnames = ['{}.jpg'.format(i) for i in range(3000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_dog, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
 
# 复制接下来1000张狗图片到 validation_dogs_dir
fnames = ['{}.jpg'.format(i) for i in range(3000, 4000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_dog, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
 
# 复制接下来1000张狗图片到 test_dogs_dir
fnames = ['{}.jpg'.format(i) for i in range(4000, 5000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_dog, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)
 
print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))

"""数据增强"""

# 对训练图像进行数据增强
train_pic_gen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.5,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
# 对测试图像进行数据增强
test_pic_gen = ImageDataGenerator(rescale=1./255)
# 利用 .flow_from_directory 函数生成训练数据
train_flow = train_pic_gen.flow_from_directory(train_dir,
                                               target_size=(224,224),
                                               batch_size=64,
                                               class_mode='categorical')
# 利用 .flow_from_directory 函数生成测试数据
test_flow = test_pic_gen.flow_from_directory(test_dir,
                                             target_size=(224,224),
                                             batch_size=64,
                                             class_mode='categorical')
print(train_flow.class_indices)


"""网络构建"""

resize = 150
model = Sequential()
# level1
model.add(Conv2D(filters=96,kernel_size=(11,11),
                 strides=(4,4),padding='valid',
                 input_shape=(resize,resize,3),
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3),
                       strides=(2,2),
                       padding='valid'))
model.add(Dropout(0.3))

# level_2
model.add(Conv2D(filters=256,kernel_size=(5,5),
                 strides=(1,1),padding='same',
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3),
                       strides=(2,2),
                       padding='valid'))
model.add(Dropout(0.3))

# layer_3
model.add(Conv2D(filters=384,kernel_size=(3,3),
                 strides=(1,1),padding='same',
                 activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=384,kernel_size=(3,3),
                 strides=(1,1),padding='same',
                 activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=356,kernel_size=(3,3),
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3),
                       strides=(2,2),padding='valid'))

# layer_4
model.add(Flatten())
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.5))

# output layer
model.add(Dense(1, activation='sigmoid'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
 
print(model.summary())

"""训练"""
 
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['acc'])
 
 
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

 
train_generator = train_datagen.flow_from_directory(
    train_dir,  # target directory
    target_size=(150, 150),  # resize图片
    batch_size=20,
    class_mode='binary'
)
 
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)
 
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break
 
hist = model.fit_generator(
    train_generator,
    steps_per_epoch=300,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=50
)

"""存储模型"""
model.save('cats_and_dogs_small_1.h5')


"""绘图"""
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
 
plt.legend()
plt.figure()
 
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.legend()
plt.show()
