import os
import time
import sys
sys.path.append('../')
import torch
from torch.utils.tensorboard import SummaryWriter
import shutil

from sklearn.metrics import confusion_matrix

from utils import plot_confusion_matrix, most_recent_folder, most_recent_folder, most_recent_weights, last_epoch, save_report, quadratic_weighted_kappa

class Trainer(object):
	"""
	Trainer class
	"""
	def __init__(self, model, criterion, optimizer, config, train_loader, valid_loader, lr_scheduler, warmup_scheduler):

		self.model = model
		self.config = config
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.criterion = criterion
		self.optimizer = optimizer
		self.lr_scheduler = lr_scheduler
		self.warmup_scheduler = warmup_scheduler

		self.batch_size = config.loader_params['bs']
		self.total_train_samples = self.batch_size*len(train_loader)
		self.total_valid_samples = self.batch_size*len(valid_loader)
		self.start_epoch = 1
		self.epochs = config.EPOCH

		self.logger_setup = self._setup_logging(config)


	def _train_epoch(self, epoch):
		"""
		Procedure for training a single epoch.

		Args:
			epoch (int): Current training epoch.
		"""
		start = time.time()
		self.model.train()
		for batch_idx, (images, labels) in enumerate(self.train_loader):

			images, labels = images.to(self.config.DEVICE), labels.to(self.config.DEVICE)

			self.optimizer.zero_grad()
			output = self.model(images)
			loss = self.criterion(output, labels)
			loss.backward()
			self.optimizer.step()

			n_iter = (epoch - 1) * len(self.train_loader) + batch_idx + 1

			last_layer = list(self.model.children())[-1]
			for name, para in last_layer.named_parameters():
				if 'weight' in name:
					self.logger_setup['writer'].add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
				if 'bias' in name:
					self.logger_setup['writer'].add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

			print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
				loss.item(),
				self.optimizer.param_groups[0]['lr'],
				epoch=epoch,
				trained_samples=self.config.loader_params['bs']*(batch_idx + 1),
				total_samples=self.total_train_samples))
		
			#update training loss for each iteration
			self.logger_setup['writer'].add_scalar('Train/loss', loss.item(), n_iter)
			
			if self.config.WARM_UP and (epoch <= self.config.WARM_EPOCH):
				self.warmup_scheduler.step()

		for name, param in self.model.named_parameters():
			layer, attr = os.path.splitext(name)
			attr = attr[1:]
			self.logger_setup['writer'].add_histogram("{}/{}".format(layer, attr), param, epoch)

		finish = time.time()
		print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

	@torch.no_grad()
	def _valid_epoch(self, epoch=0, tb=True):
		
		"""
		Procedudre for validating a single epoch.

		Args:
			epoch (int): Current training epoch.
		"""
		start = time.time()
		self.model.eval()

		valid_loss = 0.0
		correct = 0.0

		all_predictions = []
		all_targets = []
		for images, labels in self.valid_loader:

			images, labels = images.to(self.config.DEVICE), labels.to(self.config.DEVICE)

			outputs = self.model(images)
			loss = self.criterion(outputs, labels)

			valid_loss += loss.item() * images.size(0)
			_, preds = outputs.max(1)
			correct += preds.eq(labels).sum()

			all_predictions.extend(preds.cpu().tolist())
			all_targets.extend(labels.cpu().tolist())

		finish = time.time()
		matrix = confusion_matrix(all_targets, all_predictions)

		fig = plot_confusion_matrix(matrix, self.config.CLASS_NAMES, normalize=True)
		fig.savefig(os.path.join(self.logger_setup['plots_dir'],'confusion_matrix_epoch_'+str(epoch)+'.png'), bbox_inches='tight')
		print("all targets",all_targets)
		print("all predictions", all_predictions)
		print("set(targets) - set(predictions)", set(all_targets)-set(all_predictions))
		save_report(all_targets, all_predictions, self.config.CLASS_NAMES, self.logger_setup['reports_dir'], epoch)
		weighted_kappa = quadratic_weighted_kappa(matrix)
		self.logger_setup['writer'].add_figure('Test/Confusion Matrix', fig, epoch)

		print('Evaluating Network.....')
		print('Validation set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
			epoch,
			valid_loss / self.total_valid_samples,
			correct.float() / self.total_valid_samples,
			finish - start))
		print()
		if tb:
			self.logger_setup['writer'].add_scalar('Test/Average loss', valid_loss / self.total_valid_samples, epoch)
			self.logger_setup['writer'].add_scalar('Test/Accuracy', correct.float() / self.total_valid_samples, epoch)
			self.logger_setup['writer'].add_scalar('Test/Quadratic weighted kappa',  weighted_kappa, epoch)

		return (correct.float() / self.total_valid_samples, weighted_kappa)

	def _setup_logging(self, config):

		logger_setup = {}

		if not os.path.exists(config.LOG_DIR):
			os.mkdir(config.LOG_DIR)
		
		if not os.path.exists(config.PLOTS_DIR):
			os.mkdir(config.PLOTS_DIR)
		
		logger_setup['plots_dir'] = os.path.join(config.PLOTS_DIR, config.MODEL+'/', config.TIME_NOW+'/')

		if not os.path.exists(os.path.join(config.PLOTS_DIR, config.MODEL+'/')):
			os.mkdir(os.path.join(config.PLOTS_DIR, config.MODEL +'/'))

		if not os.path.exists(logger_setup['plots_dir']):
			os.mkdir(logger_setup['plots_dir'])

		logger_setup['reports_dir'] = os.path.join(config.REPORTS_DIR, config.MODEL+'/', config.TIME_NOW+'/')

		if not os.path.exists(os.path.join(config.REPORTS_DIR, config.MODEL+'/')):
			os.mkdir(os.path.join(config.REPORTS_DIR, config.MODEL+'/'))

		if not os.path.exists(logger_setup['reports_dir']):
			os.mkdir(logger_setup['reports_dir'])

		if config.RESUME:
			recent_folder = most_recent_folder(os.path.join(config.CHECKPOINT_PATH, config.MODEL), fmt=config.DATE_FORMAT)
			if not recent_folder:
				raise Exception('no recent folder was found')

			checkpoint_path = os.path.join(config.CHECKPOINT_PATH, config.MODEL, recent_folder)

		else:
			checkpoint_path = os.path.join(config.CHECKPOINT_PATH, config.MODEL, config.TIME_NOW)

		if not os.path.exists(checkpoint_path):
			os.makedirs(checkpoint_path)

		logger_setup['checkpoint_file'] = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
		
		logger_setup['writer'] = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, config.MODEL, config.TIME_NOW))
		
		shutil.copy('config.py', os.path.join(config.LOG_DIR, config.MODEL, config.TIME_NOW))

		if config.RESUME:
			recent_weights_file = most_recent_weights(os.path.join(config.CHECKPOINT_PATH, config.MODEL, recent_folder))
			if not recent_weights_file:
				raise Exception('no recent weights file were found')
			weights_path = os.path.join(config.CHECKPOINT_PATH, config.MODEL, recent_folder, recent_weights_file)
			print('loading weights file {} to resume training.....'.format(weights_path))
			self.model.load_state_dict(torch.load(weights_path))

			resume_epoch = last_epoch(os.path.join(config.CHECKPOINT_PATH, config.MODEL, recent_folder))
		else:
			resume_epoch=0
		logger_setup['resume_epoch'] = resume_epoch
		
		return logger_setup

	def train(self):
		"""
		Full training logic
		"""
		best_acc = 0.0
		best_kappa = 0.0
		for epoch in range(self.start_epoch, self.epochs + 1):

			if self.config.WARM_UP and epoch > self.config.WARM_EPOCH:
				self.lr_scheduler.step()

			elif not self.config.WARM_UP:
				self.lr_scheduler.step()

			if self.config.RESUME:
				if epoch <= self.logger_setup['resume_epoch']:
					continue
			
			self._train_epoch(epoch)
			acc, kappa = self._valid_epoch(epoch)
			
			

			if best_acc < acc or best_kappa < kappa:
				weights_path = self.logger_setup['checkpoint_file'].format(net=self.config.MODEL, epoch=epoch, type='best')
				print('saving weights file to {}'.format(weights_path))
				torch.save(self.model.state_dict(), weights_path)
				best_acc = acc
				best_kappa = kappa
		
		self.logger_setup['writer'].close()
				
	




