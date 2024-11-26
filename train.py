import argparse
import time

import numpy as np
import torch as torch

from dataset import Dataset
from evaluator import evaluate, cal_return_ratios
from model import select_model
from seed import set_seeds


class Trainer:
    def __init__(self, model, optimizer, lr_scheduler,
                 data_feature, train_dataloader, eval_dataloader, test_dataloader):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.data_feature = data_feature
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.scaler = data_feature['scaler']

        self.best_score = float('-inf')
        self.best_valid_perf = None
        self.best_test_perf = None
        self.best_weights = self.model.state_dict().copy()
        self.val_step = 5

    def train(self, epochs):
        train_loss = []
        step = 0
        for epoch in range(epochs):
            start_time = time.time()
            print(f"\nEpoch {epoch} training..., lr: {self.optimizer.param_groups[0]['lr']}")
            for data in self.train_dataloader:
                self.model.train()
                step += 1
                data.to_tensor(device)
                self.optimizer.zero_grad()
                loss = self.model.cal_train_loss(data)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
                self.lr_scheduler.step()

                if step % self.val_step == 0:
                    self.validate(self.eval_dataloader, if_record=True)
            print(f"Epoch {epoch} training finished, time: {time.time() - start_time}")

    @torch.no_grad()
    def validate(self, data_loader, if_record=False):
        self.model.eval()
        with torch.no_grad():
            masks = []
            pred_ratios = []
            true_ratios = []
            for data in data_loader:
                data.to_tensor(device)
                pred = self.model.predict(data)

                pred = self.scaler.inverse_transform(pred)
                data['x'] = self.scaler.inverse_transform(data['x'])
                data['y'] = self.scaler.inverse_transform(data['y'])

                data['mask'] = data['mask'].min(dim=1)[0].unsqueeze(1)
                masks.append(data['mask'].cpu().numpy())
                last_price = data['x'][:, -1, :, -1]
                data['mask'] = data['mask'].min(dim=1)[0].unsqueeze(1)
                pred_ratio, true_ratio = cal_return_ratios(pred, data['y'], last_price)
                pred_ratios.append(pred_ratio.cpu().numpy())
                true_ratios.append(true_ratio.cpu().numpy())
            masks = np.concatenate(masks, axis=0).squeeze()
            pred_ratios = np.concatenate(pred_ratios, axis=0).squeeze().transpose(1, 0)
            true_ratios = np.concatenate(true_ratios, axis=0).squeeze().transpose(1, 0)
            masks = masks.transpose(1, 0)
            perf = evaluate(pred_ratios, true_ratios, masks)

            if perf['total'] > self.best_score and if_record:
                print(f"New best performance")
                self.best_score = perf['total']
                self.best_valid_perf = perf
                self.best_test_perf = perf
                self.best_weights = self.model.state_dict().copy()
            return perf['total']

    def test(self):
        print("\n Final test performance......")
        self.model.load_state_dict(self.best_weights)
        start_time = time.time()
        _ = self.validate(self.test_dataloader)
        print(f"Best performance: {self.best_test_perf}")
        print(f"Test time: {time.time() - start_time}")


if __name__ == '__main__':
    set_seeds(42)
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

    args = argparse.Namespace()
    args.market_name = "NASDAQ"
    args.input_window = 16
    args.output_window = 1
    args.output_dim = 1
    args.scaler_type = "none"

    if args.market_name == 'NASDAQ':
        args.batch_size = 32
    else:
        args.batch_size = 16

    args.learning_rate = 0.001
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = Dataset(args)
    train_dataloader, eval_dataloader, test_dataloader = dataset.get_data()
    data_feature = dataset.get_data_feature()

    model = select_model('SDL', args, data_feature, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    trainer = Trainer(model, optimizer, lr_scheduler,
                      data_feature, train_dataloader, eval_dataloader, test_dataloader)
    trainer.train(epochs=50)
    trainer.test()
