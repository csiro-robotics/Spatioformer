from scripts.spatioformer import SpatioformerModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import datetime
import os
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pickle
import argparse


class MyDataset(Dataset):

    def __init__(self, pickle_dir='data_to_release/samples_to_release.pkl'):
        with open(pickle_dir, 'rb') as f:         
            self.imgs = pickle.load(f)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        row = self.imgs.loc[idx]
        array, lon_4326, lat_4326, richness = row['Image'], row['Longitude'], row['Latitude'], row['Richness']

        return torch.from_numpy(array.astype('float32')), richness.astype('float32'), torch.from_numpy(lon_4326.astype('float32')), torch.from_numpy(lat_4326.astype('float32'))

    
def get_dataloaders(
        batch_size=2048,
        num_workers=os.cpu_count(),
        split_file='data_to_release/split_to_release.pkl',
        ):
    
    dataset = MyDataset()
    
    with open(split_file, 'rb') as f:         
        split = pickle.load(f)

    train_indices = split['train']
    val_indices = split['val']
    test_indices = split['test']
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
    val_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, sampler=val_sampler)
    test_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, sampler=test_sampler)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Spatioformer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-size', default=9, type=int, help='input image size', required=False)
    parser.add_argument('--epochs', default=100, type=int, help='number of epoches to train', required=False)
    parser.add_argument('--learning-rate', default=0.001, type=float, help='learning rate for training', required=False)
    parser.add_argument('--weight-decay', default=0.0001, type=float, help='weight decay for training', required=False)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on {device}')
    net = SpatioformerModel(patchsize=args.input_size).to(device)
    train_loader, val_loader, test_loader = get_dataloaders()

    train_loss = 0.
    val_loss = 0.
    best_val_loss = np.inf
    num_steps_to_val = 10
    time = datetime.datetime.now()
    
    model_folder = f'models/spatioformer/{time}/' if args.input_size == 9 else f'models/spatioformer/input_size_{args.input_size}/{time}/' 
    crop_start = int(5 - (args.input_size + 1) / 2)
    crop_end = int(4 + (args.input_size + 1) / 2)

    ###
    # Define loss function, optimization method, and learning rate schedule
    ###

    criterion = nn.MSELoss()
    params = [x for x in net.parameters()]
    optimizer = torch.optim.Adam(params=params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100)

    ###
    # Training starts
    ###

    for epoch_counter in range(args.epochs):
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        train_loss = 0.
        for step_counter, (array, richness, lon, lat) in enumerate(train_loader):
            array = array.to(device)[:, crop_start:crop_end, crop_start:crop_end, :]
            richness = richness.to(device)
            lon = lon.to(device)[:, crop_start:crop_end, crop_start:crop_end]
            lat = lat.to(device)[:, crop_start:crop_end, crop_start:crop_end]
            net.train()
            net.zero_grad()
            predicted = net(array, lon, lat).squeeze(-1)
            mse = criterion(predicted, richness)
            loss = torch.sqrt(mse)
            loss.backward()  # Back propagation
            optimizer.step()  # Update network weights
            scheduler.step(epoch_counter + step_counter / len(train_loader))
            train_loss += loss

            ###
            # Validation starts
            ###

            if (step_counter + 1) % num_steps_to_val == 0:
                net.eval()
                for val_step_counter, (array, richness, lon, lat) in enumerate(val_loader):
                    array = array.to(device)[:, crop_start:crop_end, crop_start:crop_end, :]
                    richness = richness.to(device)
                    lon = lon.to(device)[:, crop_start:crop_end, crop_start:crop_end]
                    lat = lat.to(device)[:, crop_start:crop_end, crop_start:crop_end]
                    predicted = net(array, lon, lat).squeeze(-1)
                    mse = criterion(predicted, richness)
                    val_loss += torch.sqrt(mse)

                print(f'Epoch: {epoch_counter + 1} | Step: {step_counter + 1} | Train loss: {round(train_loss.item() / num_steps_to_val, 2)} | Validation loss: {round(val_loss.item() / len(val_loader), 2)} | Best validation loss: {round(best_val_loss, 2)}')

                if val_loss.item() / len(val_loader) < best_val_loss:
                    best_val_loss = val_loss.item() / len(val_loader)
                    torch.save(net.state_dict(), f'{model_folder}/model.pth')

                train_loss = 0.
                val_loss = 0.
