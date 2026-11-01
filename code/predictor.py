import torch
import torch.nn as nn
import numpy as np

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not found. Install with `pip install xgboost`.")

class XGBoostPredictor:
    """
    XGBoost Predictor wrapper.
    Used to replace the final MLP classifier.
    """
    def __init__(self, params=None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed.")

        self.params = params if params else {
            'max_depth': 6,
            'eta': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'nthread': 4
        }
        self.model = None
        self.num_rounds = 100

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Args:
            X: numpy array [N, features]
            y: numpy array [N]
        """
        dtrain = xgb.DMatrix(X, label=y)
        evals = []
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, 'train'), (dval, 'eval')]

        self.model = xgb.train(
            self.params,
            dtrain,
            self.num_rounds,
            evals=evals,
            early_stopping_rounds=10 if evals else None,
            verbose_eval=False
        )

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

class MLPPredictor(nn.Module):
    """
    Standard MLP Predictor (PyTorch).
    """
    def __init__(self, input_dim, hidden_dim=32, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class SklearnPredictor:
    """
    Wrapper for Scikit-learn predictors (LR, SVM, RF).
    """
    def __init__(self, predictor_type='lr', params=None):
        self.type = predictor_type
        self.params = params if params else {}
        self.model = None

        if self.type == 'lr':
            self.model = LogisticRegression(**self.params)
        elif self.type == 'svm':
            # Ensure probability=True for AUC calculation
            if 'probability' not in self.params:
                self.params['probability'] = True
            self.model = SVC(**self.params)
        elif self.type == 'rf':
            self.model = RandomForestClassifier(**self.params)
        else:
            raise ValueError(f"Unknown sklearn predictor: {self.type}")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)

class HybridPredictor:
    """
    Wrapper to switch between MLP, XGBoost, and Sklearn predictors.
    """
    def __init__(self, predictor_type='xgboost', input_dim=64, device='cpu', predictor_params=None):
        self.type = predictor_type
        self.device = device
        self.predictor_params = predictor_params or {}

        if self.type == 'mlp':
            self.model = MLPPredictor(input_dim=input_dim).to(device)
        elif self.type == 'xgboost':
            # Pass params into XGBoost; allow overriding num_rounds
            xgb_params = {k: v for k, v in self.predictor_params.items() if k != 'num_rounds'}
            self.model = XGBoostPredictor(params=xgb_params)
            if 'num_rounds' in self.predictor_params:
                self.model.num_rounds = int(self.predictor_params['num_rounds'])
        elif self.type in ['lr', 'svm', 'rf']:
            self.model = SklearnPredictor(predictor_type=self.type, params=self.predictor_params)
        else:
            # Fallback
             self.type = 'xgboost'
             xgb_params = {k: v for k, v in self.predictor_params.items() if k != 'num_rounds'}
             self.model = XGBoostPredictor(params=xgb_params)
             if 'num_rounds' in self.predictor_params:
                 self.model.num_rounds = int(self.predictor_params['num_rounds'])

    def parameters(self):
        if self.type == 'mlp':
            return self.model.parameters()
        return [] # Non-MLP has no PyTorch parameters

    def train(self):
        if self.type == 'mlp':
            self.model.train()

    def eval(self):
        if self.type == 'mlp':
            self.model.eval()

    def __call__(self, x):
        # Forward pass for MLP
        if self.type == 'mlp':
            return self.model(x)
        else:
            raise NotImplementedError(f"{self.type} cannot be called directly in a PyTorch computation graph.")

    def fit_xgboost(self, embeddings, labels, val_embeddings=None, val_labels=None):
        # Generic fit method for non-MLP predictors (XGBoost/Sklearn)
        if self.type == 'mlp':
            return

        # Detach and move to CPU numpy
        X = embeddings.detach().cpu().numpy()
        y = labels.detach().cpu().numpy()

        if self.type == 'xgboost':
            X_val, y_val = None, None
            if val_embeddings is not None:
                X_val = val_embeddings.detach().cpu().numpy()
                y_val = val_labels.detach().cpu().numpy()
            self.model.fit(X, y, X_val, y_val)
        else: # Sklearn
             self.model.fit(X, y)

    def predict_xgboost(self, embeddings):
        # Generic predict method for non-MLP predictors
        if self.type == 'mlp':
            return None

        X = embeddings.detach().cpu().numpy()

        if self.type == 'xgboost':
            preds = self.model.predict(X)
        else: # Sklearn
            preds = self.model.predict_proba(X)

        return torch.tensor(preds).to(self.device)
