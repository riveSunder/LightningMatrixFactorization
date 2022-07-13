import os

import numpy as np 

import torch
import torch.nn.functional as F

import pytorch_lightning as pl

class LightningMatrixFactorization(pl.LightningModule):
    
    def __init__(self, number_of_books, number_of_users, **kwargs):
        super().__init__()
        
        self.learning_rate = kwargs["lr"]\
                if "lr" in kwargs.keys() else 1e-3
        self.number_of_books = number_of_books
        self.number_of_users = number_of_users
        self.embed_dim = kwargs["embed_dim"]\
                if "embed_dim" in kwargs.keys() else 32
        
        self.embed_users = torch.nn.Embedding(\
                self.number_of_users, self.embed_dim)
        self.embed_books = torch.nn.Embedding(\
                self.number_of_books, self.embed_dim)
    
    def forward(self, users, books):
        
        embedded_books = self.embed_books(books)
        embedded_users = self.embed_users(users)
        
        predicted = torch.sum(\
                torch.multiply(embedded_users, embedded_books),\
                dim=-1)
        
        return predicted
    
    def training_step(self,batch, batch_idx):
        
        users = batch[0] 
        books = batch[1]
        ratings = batch[2]
        
        embedded_books = self.embed_books(books)
        embedded_users = self.embed_users(users)
        
        predicted = torch.sum(\
                torch.multiply(embedded_users, embedded_books), \
                dim=-1)
        
        loss = F.mse_loss(predicted, ratings)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        users = batch[0] 
        books = batch[1]
        val_ratings = batch[2]
        
        embedded_books = self.embed_books(books)
        embedded_users = self.embed_users(users)
        
        val_predicted = torch.sum(\
                torch.multiply(embedded_users, embedded_books),\
                dim=-1)
        
        val_loss = F.mse_loss(val_predicted, val_ratings)
        self.log("val_loss", val_loss)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),\
                lr=self.learning_rate)
        return optimizer
