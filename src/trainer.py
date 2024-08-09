import time
from typing import Optional, Dict
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .utils import TensorboardWriter, AverageMeter, save_checkpoint, \
    clip_gradient, adjust_learning_rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(
        self,
        num_epochs: int,
        start_epoch: int,
        train_loader: DataLoader,
        model: nn.Module,
        loss_function: nn.Module,
        optimizer,
        lr_decay: float,
        grad_clip = Optional[None],
        print_freq: int = 100,
        checkpoint_path: Optional[str] = None,
        checkpoint_basename: str = 'checkpoint',
        tensorboard: bool = False,
        log_dir: Optional[str] = None
    ) -> None:
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self.train_loader = train_loader

        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_decay = lr_decay
        self.print_freq = print_freq
        self.grad_clip = grad_clip

        self.checkpoint_path = checkpoint_path
        self.checkpoint_basename = checkpoint_basename

        # setup visualization writer instance
        self.writer = TensorboardWriter(log_dir, tensorboard)
        self.len_epoch = len(self.train_loader)

    def train(self, epoch: int) -> None:
        """
        Train an epoch

        Parameters
        ----------
        epoch : int
            Current number of epoch
        """
        self.model.train()  # training mode enables dropout

        batch_time = AverageMeter()  # forward prop. + back prop. time per batch
        data_time = AverageMeter()  # data loading time per batch
        losses = AverageMeter(tag='loss', writer=self.writer) 
        accs = AverageMeter(tag='acc', writer=self.writer) 

        start = time.time()

        # batches
        for i, batch in enumerate(self.train_loader):
            data_time.update(time.time() - start)
            input_tensor, labels = batch

            input_tensor = input_tensor.to(device)
            labels = labels.squeeze(1).to(device)  # (batch_size)
            scores = self.model(input_tensor)
            loss = self.loss_function(scores, labels)

            # backward
            self.optimizer.zero_grad()
            loss.backward()

            # clip gradients
            if self.grad_clip is not None:
                clip_gradient(self.optimizer, self.grad_clip)

            # update weights
            self.optimizer.step()

            # find accuracy
            _, predictions = scores.max(dim = 1)
            correct_predictions = torch.eq(predictions, labels).sum().item()
            accuracy = correct_predictions / labels.size(0)

            # set step for tensorboard
            step = (epoch - 1) * self.len_epoch + i
            self.writer.set_step(step=step, mode='train')

            # keep track of metrics
            batch_time.update(time.time() - start)
            losses.update(loss.item(), labels.size(0))
            accs.update(accuracy, labels.size(0))

            start = time.time()

            # print training status
            if i % self.print_freq == 0:
                print(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch, i, len(self.train_loader),
                        batch_time = batch_time,
                        data_time = data_time,
                        loss = losses,
                        acc = accs
                    )
                )

    def run_train(self):
        start = time.time()

        # epochs
        for epoch in range(self.start_epoch, self.num_epochs):
            # trian an epoch
            self.train(epoch=epoch)

            # time per epoch
            epoch_time = time.time() - start
            print('Epoch: [{0}] finished, time consumed: {epoch_time:.3f}'.format(epoch, epoch_time=epoch_time))

            # decay learning rate every epoch
            adjust_learning_rate(self.optimizer, self.lr_decay)

            # save checkpoint
            if self.checkpoint_path is not None:
                save_checkpoint(
                    epoch = epoch,
                    model = self.model,
                    model_name = self.model_name,
                    optimizer = self.optimizer,
                    dataset_name = self.dataset_name,
                    word_map = self.word_map,
                    checkpoint_path = self.checkpoint_path,
                    checkpoint_basename = self.checkpoint_basename
                )

            start = time.time()