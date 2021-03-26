import time

class Trainer(object):
	"""
	Trainer class
	"""
	def __init__(self, model, criterion, optimizer, config, train_loader, valid_loader):
		self.model = model
		self.config = config
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.criterion = criterion
		self.optimizer = optimizer

		self.batch_size = config.loader_params['bs']
		self.total_train_samples = self.batch_size*len(train_loader)
		self.total_valid_samples = self.batch_size*len(valid_loader)
		self.start_epoch = 1
		self.epochs = config.EPOCH


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

			print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
				loss.item(),
				self.optimizer.param_groups[0]['lr'],
				epoch=epoch,
				trained_samples=len(images)*(batch_idx + 1),
				total_samples=self.total_train_samples))
		
		finish = time.time()
		print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

	def _valid_epoch(self, epoch):
		
		"""
		Procedudre for validating a single epoch.

		Args:
			epoch (int): Current training epoch.
		"""
		start = time.time()
		self.model.eval()

		test_loss = 0.0
		correct = 0.0

		for images, labels in self.valid_loader:

			images, labels = images.to(self.config.DEVICE), labels.to(self.config.DEVICE)

			outputs = self.model(images)
			loss = self.criterion(outputs, labels)

			test_loss += loss.item() * images.size(0)
			_, preds = outputs.max(1)
			correct += preds.eq(labels).sum()

		finish = time.time()

		print('Evaluating Network.....')
		print('Validation set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
			epoch,
			test_loss / self.total_valid_samples,
			correct.float() / self.total_valid_samples,
			finish - start))
		print()

		return correct.float() / self.total_valid_samples

	def train(self):
		"""
		Full training logic
		"""
		
		for epoch in range(self.start_epoch, self.epochs + 1):
			self._train_epoch(epoch)
			acc = self._valid_epoch(epoch)





