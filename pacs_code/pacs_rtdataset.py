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

class rtPACS(Dataset):
	def __init__(self, test_domain, num_samples=20, num_domains=3, transform=None):
		# assert phase in ['train', 'val', 'test']
		self.domain_list = ['art_painting', 'photo', 'cartoon', 'sketch']
		self.domain_list.remove(test_domain)
		self.num_domains = num_domains
		self.num_samples = num_samples
		assert self.num_domains <= len(self.domain_list)

		self.sample_list = []

		self.infer_imgs = []
		self.infer_labels = []

		# self.num_imgs = []
		for i in range(len(self.domain_list)):
			f = open('../files/' + self.domain_list[i] + '_train_kfold.txt', 'r')
			lines = f.readlines()
			samples = {}
			# domain_imgs = {}
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				label = int(label) - 1
				if label not in samples.keys():
					samples[label] = []
				samples[label].append(data_path + img)
			self.sample_list.append(samples)

			# pdb.set_trace()

			for i in range(len(samples.keys())):
				self.infer_imgs = self.infer_imgs + samples[i][:num_samples]   # 20 samples for center feature during test
				self.infer_labels = self.infer_labels + [i] * num_samples

		# pdb.set_trace()

	def reset(self, phase, transform=None):
		# pdb.set_trace()
		self.phase = phase
		self.transform = transform
		if phase == 'train':
			self.img_list = []
			self.label_list = []
			for i in range(self.num_domains):
				for j in range(7):
					# pdb.set_trace()
					np.random.shuffle(self.sample_list[i][j])
					self.img_list = self.img_list + self.sample_list[i][j][:self.num_samples]
					self.label_list = self.label_list + [j] * self.num_samples

		else:
			self.img_list = self.infer_imgs
			self.label_list = self.infer_labels

		# pdb.set_trace()
		assert len(self.img_list)==len(self.label_list)

	def __getitem__(self, item):
		# image
		image = Image.open(self.img_list[item]).convert('RGB')  # (C, H, W)
		# print(np.array(image).shape)
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