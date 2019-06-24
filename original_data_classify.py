import os
import shutil  # 复制文件

# 原始目录所在的路径
# 数据集未压缩
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

