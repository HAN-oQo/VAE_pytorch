import torch
import os 
import shutil
import stat

data_path = '../data/celeba'


def preprocess(data_path):

    
    src_dir = os.path.join(data_path, 'images')
    img_list = os.listdir(src_dir)
    print("total image length : {}".format(len(img_list)))
    
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    

    count = 0 
    for i in range(len(img_list)):
        filename = img_list[i]
        jpgfile = os.path.join(src_dir, filename)
        # os.chmod(jpgfile, stat.S_IRWXU|stat.S_IRWXG|stat.S_IRWXO)
        if (i+1) <= 2000:
            if not os.path.exists(os.path.join(test_dir, filename)):
                shutil.copy(jpgfile, test_dir)

        else:
            if not os.path.exists(os.path.join(train_dir, filename)):
                shutil.copy(jpgfile, train_dir)
    
    train = os.listdir(train_dir)
    test = os.listdir(test_dir)

    print("train dataset length: {}".format(len(train)))
    print("test dataset length: {}".format(len(test)))

preprocess(data_path)
