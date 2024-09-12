import torch
import torch.distributed as dist

import numpy as np
import os
import logging
import h5py
import sys

def dist_average(metrics, count=1.):
	tensor = torch.tensor(list(metrics) + [1], device='cuda', dtype=torch.float32)
	tensor *= count
	torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
	return (tensor[:-1] / tensor[-1]).cpu().numpy().tolist()

class data_prefetcher():
	def __init__(self, loader):
		self.loader = iter(loader)
		self.stream = torch.cuda.Stream()
		self.preload()

	def preload(self):
		try:
			self.next_audio_amp, self.next_audio_raw, \
			self.next_radio_amp = next(self.loader) #self.next_audio_phase, , self.next_radio_phase
		except StopIteration:
			self.next_audio_amp = None
			# self.next_audio_phase = None
			self.next_audio_raw = None
			self.next_radio_amp = None
			# self.next_radio_phase = None
			return
		with torch.cuda.stream(self.stream):
			self.next_audio_amp = self.next_audio_amp.cuda(non_blocking=True)
			# self.next_audio_phase = self.next_audio_phase.cuda(non_blocking=True)
			self.next_audio_raw = self.next_audio_raw.cuda(non_blocking=True)
			self.next_radio_amp = self.next_radio_amp.cuda(non_blocking=True)
			# self.next_radio_phase = self.next_radio_phase.cuda(non_blocking=True)

	def next(self):
		torch.cuda.current_stream().wait_stream(self.stream)
		next_audio_amp = self.next_audio_amp
		# next_audio_phase = self.next_audio_phase
		next_audio_raw = self.next_audio_raw
		next_radio_amp = self.next_radio_amp
		# next_radio_phase = self.next_radio_phase
		self.preload()
		return next_audio_amp, next_audio_raw, next_radio_amp #next_audio_phase, , next_radio_phase


class AverageMeter(object):
#Computes and stores the average and current value
	def __init__(self):
		self.initialized = False
		self.val = None
		self.avg = None
		self.sum = None
		self.count = None

	def initialize(self, val, weight):
		self.val = val
		self.avg = val
		self.sum = val*weight
		self.count = weight
		self.initialized = True

	def update(self, val, weight=1):
		val = np.asarray(val)
		if not self.initialized:
			self.initialize(val, weight)
		else:
			self.add(val, weight)

	def add(self, val, weight):
		self.val = val
		self.sum += val * weight
		self.count += weight
		self.avg = self.sum / self.count

	def value(self):
		if self.val is None:
			return 0.
		else:
			return self.val.tolist()

	def average(self):
		if self.avg is None:
			return 0.
		else:
			return self.avg.tolist()

def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
	"""Write dataset to hdf5.
	Args:
		hdf5_name (str): Hdf5 dataset filename.
		hdf5_path (str): Dataset path in hdf5.
		write_data (ndarray): Data to write.
		is_overwrite (bool): Whether to overwrite dataset.
	"""
	# convert to numpy array
	write_data = np.array(write_data)

	# check folder existence
	folder_name, _ = os.path.split(hdf5_name)
	if not os.path.exists(folder_name) and len(folder_name) != 0:
		os.makedirs(folder_name)

	# check hdf5 existence
	if os.path.exists(hdf5_name):
		# if already exists, open with r+ mode
		hdf5_file = h5py.File(hdf5_name, "r+")
		# check dataset existence
		if hdf5_path in hdf5_file:
			if is_overwrite:
				logging.warning(
					"Dataset in hdf5 file already exists. " "recreate dataset in hdf5."
				)
				hdf5_file.__delitem__(hdf5_path)
			else:
				logging.error(
					"Dataset in hdf5 file already exists. "
					"if you want to overwrite, please set is_overwrite = True."
				)
				hdf5_file.close()
				sys.exit(1)
	else:
		# if not exists, open with w mode
		hdf5_file = h5py.File(hdf5_name, "w")

	# write data to hdf5
	hdf5_file.create_dataset(hdf5_path, data=write_data)
	hdf5_file.flush()
	hdf5_file.close()

def read_hdf5(hdf5_name, hdf5_path):
	"""Read hdf5 dataset.
	Args:
		hdf5_name (str): Filename of hdf5 file.
		hdf5_path (str): Dataset name in hdf5 file.
	Return:
		any: Dataset values.
	"""
	if not os.path.exists(hdf5_name):
		logging.error(f"There is no such a hdf5 file ({hdf5_name}).")
		sys.exit(1)

	hdf5_file = h5py.File(hdf5_name, "r")

	if hdf5_path not in hdf5_file:
		logging.error(f"There is no such a data in hdf5 file. ({hdf5_path})")
		sys.exit(1)

	hdf5_data = hdf5_file[hdf5_path][()]
	hdf5_file.close()

	return hdf5_data
