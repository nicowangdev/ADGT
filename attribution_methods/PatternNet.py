import torch
from tqdm import tqdm
from attribution_methods.pytorch_patternNet import PatternNetSignalEstimator


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SignalEstimator:
    def __init__(self, model):
        self.model = model
        self.signal_estimator = PatternNetSignalEstimator(self.model)

    def train_explain(self, train_dataloader):
        with torch.no_grad():
            for batch, _ in tqdm(train_dataloader):
                self.signal_estimator.update_E(batch.to(device))

        self.signal_estimator.get_patterns()

    def __call__(self, batch):
        signal = self.signal_estimator.get_signal(batch.to(device))
        return signal


class Explainer:
    def __init__(self, model, train_dataloader):
        self.model = model
        self.signal_estimator = SignalEstimator(model)
        self.signal_estimator.train_explain(train_dataloader)

    def get_pattern(self, batch):
        signal = self.signal_estimator(batch)
        return signal

