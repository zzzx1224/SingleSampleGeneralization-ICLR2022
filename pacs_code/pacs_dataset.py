import numpy as np 
import os
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
# from utils import get_transform
import pdb
import random
import torch
import time
import cv2

# data_path = '../kfold/'
data_path = '../224kfold/'

class PACS(Dataset):
	def __init__(self, test_domain, num_domains=3, transform=None):
		# assert phase in ['train', 'val', 'test']
		self.domain_list = ['art_painting', 'photo', 'cartoon', 'sketch']
		self.domain_list.remove(test_domain)
		self.num_domains = num_domains
		assert self.num_domains <= len(self.domain_list)

		self.train_img_list = []
		self.train_label_list = []

		# self.num_imgs = []
		for i in range(len(self.domain_list)):
			f = open('../files/' + self.domain_list[i] + '_train_kfold.txt', 'r')
			lines = f.readlines()
			train_domain_imgs = []
			train_domain_labels = []
			# domain_imgs = {}
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				train_domain_imgs.append(data_path + img)
				train_domain_labels.append(int(label)-1)
			self.train_img_list.append(train_domain_imgs)
			self.train_label_list.append(train_domain_labels)
			# self.num_imgs.append(len(train_domain_imgs))
		# pdb.set_trace()

		self.val_img_list = []
		self.val_label_list = []
		self.test_img_list = []
		self.test_label_list = []
		# self.transform = transform
		# self.meta_test_domain = np.random.randint(len(self.domain_list))

		seed = 777

		# elif phase == 'val':
		self.domain_list.append(test_domain)
		# pdb.set_trace()
		for i in range(len(self.domain_list)):
			f = open('../files/' + self.domain_list[i] + '_crossval_kfold.txt', 'r')
			lines = f.readlines()

			val_domain_imgs = []
			val_domain_labels = []

			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				# self.val_img_list.append(data_path + img)
				# self.val_label_list.append(int(label)-1)
				val_domain_imgs.append(data_path + img)
				val_domain_labels.append(int(label)-1)
			np.random.seed(seed)
			np.random.shuffle(val_domain_imgs)
			np.random.seed(seed)
			np.random.shuffle(val_domain_labels)
			self.val_img_list.append(val_domain_imgs)
			self.val_label_list.append(val_domain_labels)
		self.domain_list.remove(test_domain)


		# else:
		f = open('../files/' + test_domain + '_test_kfold.txt', 'r')
		lines = f.readlines()
		for line in lines:
			[img, label] = line.strip('\n').split(' ')
			self.test_img_list.append(data_path + img)
			self.test_label_list.append(int(label)-1)

		# seed = 777
		# pdb.set_trace()
		np.random.seed(seed)
		np.random.shuffle(self.test_img_list)
		np.random.seed(seed)
		np.random.shuffle(self.test_label_list)

		# pdb.set_trace()

	def reset(self, phase, domain_id, transform=None):
		# pdb.set_trace()
		self.phase = phase
		if phase == 'train':
			self.transform = transform
			self.img_list = self.train_img_list[domain_id]
			self.label_list = self.train_label_list[domain_id]
			# pdb.set_trace()

		elif phase == 'val':
			self.transform = transform
			self.img_list = self.val_img_list[domain_id]
			self.label_list = self.val_label_list[domain_id]

		elif phase == 'test':
			self.transform = transform
			self.img_list = self.test_img_list
			self.label_list = self.test_label_list

		# pdb.set_trace()
		assert len(self.img_list)==len(self.label_list)

	def __getitem__(self, item):
		# image
		image = Image.open(self.img_list[item]).convert('RGB')  # (C, H, W)
		# image = image.resize((224, 224))
		# image = cv2.imread(self.img_list[item])[::-1]
		# pdb.set_trace()
		if self.transform is not None:
			image = self.transform(image)

		label = self.label_list[item]
		# return image and label
		# return image, self.label_list[item]
		return image, label

	def __len__(self):
		return len(self.img_list)