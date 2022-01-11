import torch
import torchvision
from torch.utils.data import DataLoader,Dataset
import os
import numpy as np
from PIL import Image

class Mnist_Dataset(Dataset):

    def __init__(self,root_dir,train=False,transform=None) :

        self.root_dir = root_dir
        if train:
            img_0_path = os.path.join(self.root_dir,'mnist_train/0')
            img_0_path_str = self.root_dir + '/mnist_train/0/'
            img_7_path = os.path.join(self.root_dir,'mnist_train/7')
            img_7_path_str = self.root_dir + '/mnist_train/7/'
        else:
            img_0_path = os.path.join(self.root_dir, 'mnist_test/0')
            img_0_path_str = self.root_dir + '/mnist_test/0/'
            img_7_path = os.path.join(self.root_dir, 'mnist_test/7')
            img_7_path_str = self.root_dir + '/mnist_test/7/'
        # Get a list of image names in the folder
        img_0_list = os.listdir(img_0_path)
        img_7_list = os.listdir(img_7_path)
        len_0 = len(img_0_list)
        len_7 = len(img_7_list)
        for i in range(len_0):
            # Form the full relative path for easy access
            img_0_list[i] = img_0_path_str + img_0_list[i]
        for j in range(len_7):
            img_7_list[j] = img_7_path_str + img_7_list[j]
        self.transform = transform
        # Construct the label corresponding to the picture, where the label 0 corresponding to picture 0 and the label 1 corresponding to picture 1
        label_0_list = [0 for i in range(len_0)]
        label_7_list = [1 for i in range(len_7)]

        self.img_list = img_0_list + img_7_list
        self.label_list = label_0_list + label_7_list

        # Correspondingly shuffles the two arrays so that random and constant cancer purity samples are generated
        state = np.random.get_state()
        np.random.shuffle(self.img_list)
        np.random.set_state(state)
        np.random.shuffle(self.label_list)

    # This method accesses the corresponding image and label by index
    def __getitem__(self, index) :
        img_path = self.img_list[index]
        label = self.label_list[index]
        img = Image.open(img_path)
        img = self.transform(img)
        return img,label

    # This method calculates the length of the dataset
    def __len__(self):
        assert len(self.img_list) == len(self.label_list)
        return len(self.img_list)

root_dir = './dataset'
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32,32)),
    torchvision.transforms.ToTensor(),
])
# get the train dataset
train_dataset = Mnist_Dataset(root_dir,True,transforms)
# get the test dataset
test_dataset = Mnist_Dataset(root_dir,False,transforms)

# batch_size = 100
bag = 100

train_dataloader = DataLoader(train_dataset,bag)
test_dataloader = DataLoader(test_dataset,bag)




