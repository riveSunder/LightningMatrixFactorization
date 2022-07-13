import os

import numpy as np 
import pandas as pd 

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from lmf.lightning_matrix_factorization import LightningMatrixFactorization

import pytorch_lightning as pl

if __name__ == "__main__":

    # magic numbers
    batch_size = 262144
    learning_rate = 1e-3
    embedding_size = 32
    max_epochs = 100
    checkpoint_every = 10
    # dataloader cpu cores, based on Kaggle GPU notebooks.
    num_workers = 16

    file_path = os.path.abspath(__file__)
    root_path = os.path.split(os.path.split(file_path)[0])[0]

    print(root_path)
    my_filepath = os.path.join(root_path, "data", "ratings.csv")
    df = pd.read_csv(my_filepath)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df[:1000000]

    print(len(df))

    test_split = int(0.2 * len(df))

    train_df = df[:-2*test_split]
    val_df = df[-2*test_split:-test_split]
    test_df = df[-test_split:]

    users = torch.tensor(train_df.user_id.values).long()
    books =  torch.tensor(train_df.book_id.values).long()
    ratings = torch.tensor(train_df.rating.values).float()
    dataset = TensorDataset(users, books, ratings)

    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    val_users = torch.tensor(val_df.user_id.values).long()
    val_books =  torch.tensor(val_df.book_id.values).long()
    val_ratings = torch.tensor(val_df.rating.values).float()
    val_dataset = TensorDataset(val_users, val_books, val_ratings)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    number_of_users = np.max(df["user_id"])+1
    number_of_books = np.max(df["book_id"])+1

    lmf = LightningMatrixFactorization(number_of_books, number_of_users)

    if torch.cuda.is_available():
        trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=max_epochs)
    else:
        trainer = pl.Trainer(max_epochs=max_epochs)

    trainer.fit(model=lmf, train_dataloaders=train_loader, val_dataloaders=val_loader)        
    # testing evaluation
    with torch.no_grad():
    
        test_users = torch.tensor(test_df.user_id.values).long()
        test_books = torch.tensor(test_df.book_id.values).long()
        test_ratings = torch.tensor(test_df.rating.values).float()
        
        lmf.eval()
        test_prediction = lmf(test_users, test_books)
        
        test_loss = F.mse_loss(test_prediction, test_ratings)
        
        test_msg = f"MSE loss for test data = {test_loss:.3} \n"
        print(test_msg)

    for hh in range(10):
        # see a few examples of predictions
        lmf.eval()
        
        my_index = np.random.randint(len(test_users))
        
        my_prediction = lmf(test_users[my_index], test_books[my_index])
        
        msg = f"Test set prediction {my_prediction}, ground truth: {test_ratings[my_index]}"
        print(msg)

