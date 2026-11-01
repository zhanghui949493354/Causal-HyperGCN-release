"""Shared configuration for the released training and evaluation pipeline."""

import torch


class Config:
    def __init__(self):
        self.output_dim = 32
        self.dropout = 0.6
        self.num_epochs = 150
        self.learning_rate = 0.0005
        self.weight_decay = 0.001
        self.seed = 42
        self.early_stopping_patience = 20
        self.early_stopping_delta = 0.0001
        self.batch_size = 256
        self.num_folds = 5
        self.negative_sampling_method = "random"
        self.kmeans_n_clusters = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
