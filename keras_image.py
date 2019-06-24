
from keras.preprocessing.image import ImageDataGenerator

# 训练样本目录和测试样本目录
train_dir = './data/train/'
test_dir = './data/validation/'
# 对训练图像进行数据增强
train_pic_gen = ImageDataGenerator(rescale=1./255,rotation_range=20,
                                   width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.5,
                                   horizontal_flip=True,fill_mode='nearest')
# 对测试图像进行数据增强
test_pic_gen = ImageDataGenerator(rescale=1./255)
# 利用 .flow_from_directory 函数生成训练数据
train_flow = train_pic_gen.flow_from_directory(train_dir, target_size=(224,224),
                                               batch_size=64, class_mode='categorical')
# 利用 .flow_from_directory 函数生成测试数据
test_flow = test_pic_gen.flow_from_directory(test_dir, target_size=(224,224),
                                             batch_size=64, class_mode='categorical')
print(train_flow.class_indices)


