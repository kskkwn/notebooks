import torch


class SupervisedTrainer(object):
    """
    manage supervised learning

    Args:
        writer(utils.MLflowWriter): writer for logging outputs to mlflow
        model(nn.Module): torch model for training
        optimizer(nn.Optimizer): torch optimizer
        train_loader(torch.DataLoader): data loader for training steps
        valid_loader(torch.DataLoader): data loader for validation steps
        nb_epochs(int): number of epochs
        device(str): e.g., "cuda:0"
    """

    def __init__(self, writer, model,
                 optimizer,
                 train_loader, valid_loader,
                 criterion,
                 nb_epochs, device):
        self.writer = writer
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.nb_epochs = nb_epochs
        self.device = device

    def _calc_epoch(self, loader, is_training):
        for X, y_gt in loader:
            X = X.to(self.device).reshape(-1, 28*28)
            y_gt = y_gt.to(self.device)

            p_y = self.model(X)
            loss = self.criterion(p_y, y_gt)

            y = torch.argmax(p_y, dim=1)
            acc = torch.sum(y == y_gt)/len(y)

            if is_training:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.writer.log_metric("loss", loss, is_training)
            self.writer.log_metric("acc", acc, is_training)

        self.writer.toMlflow(step=self.epoch_i)

    def train(self):
        """
        start training loop

        Return:
            train_loss(float): training loss
            valid_loss(float): validation loss
            train_acc(float): training accuracy
            valid_acc(float): validation accuracy
        """
        for self.epoch_i in range(self.nb_epochs):
            self.model.train()
            self._calc_epoch(self.train_loader, is_training=True)
            self.model.eval()
            self._calc_epoch(self.valid_loader, is_training=False)

            self.writer.log_torch_model(self.model, self.epoch_i)
