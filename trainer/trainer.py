import numpy as np
import torch
from base import BaseTrainer
from utils import MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.dataset = data_loader.polyvore_dataset
        self.len_epoch = len(self.data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (item_indices, embeddings, input_mask, target_mask, fake_embeddings) in enumerate(self.data_loader):
            n = item_indices.shape[0]
            item_indices = item_indices.to(self.device)
            embeddings = embeddings.to(self.device)
            input_mask = input_mask.to(self.device)
            target_mask = target_mask.to(self.device)
            target_cls = torch.cat((torch.ones(n), torch.zeros(n)), dim=0).long().to(self.device)
            fake_embeddings = fake_embeddings.to(self.device)

            # Forward and backward
            self.optimizer.zero_grad()
            output, output_2 = self.model(embeddings, input_mask, target_mask, fake_embeddings, self.data_loader)
            loss = self.criterion(output, output_2, item_indices, embeddings, target_mask, target_cls, self.dataset)
            loss.backward()
            self.optimizer.step()

            # Track loss
            self.train_metrics.update('loss', loss.item(), n)

            # Track metrics
            predicted_target_indices = self.data_loader.query_top_items(output, self.device, 100)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(predicted_target_indices, item_indices, target_mask), n)

            # Log loss
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
            
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch()
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log


    def _valid_epoch(self):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for _, (item_indices, embeddings, input_mask, target_mask, fake_embeddings) in enumerate(self.valid_data_loader):
                n = item_indices.shape[0]

                item_indices = item_indices.to(self.device)
                embeddings = embeddings.to(self.device)
                input_mask = input_mask.to(self.device)
                target_mask = target_mask.to(self.device) 
                target_cls = torch.cat((torch.ones(n), torch.zeros(n)), dim=0).long().to(self.device)
                fake_embeddings = fake_embeddings.to(self.device)

                # Get output and loss
                output, output_2 = self.model(embeddings, input_mask, target_mask, fake_embeddings, None)
                loss = self.criterion(output, output_2, item_indices, embeddings, target_mask, target_cls, self.dataset)

                # Track loss
                self.valid_metrics.update('loss', loss.item(), n)

                # Track metrics
                predicted_target_indices = self.valid_data_loader.query_top_items(output, self.device, 100)
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(predicted_target_indices, item_indices, target_mask), n)

        return self.valid_metrics.result()


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
