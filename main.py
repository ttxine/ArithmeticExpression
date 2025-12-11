import argparse
import csv
import dataclasses
import os
import sys

import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             average_precision_score, fbeta_score)
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from features import FeatureExtractor
from mlp import MultilayerPerceptron


class SySeVRDataset(Dataset):
    def __init__(self, path: str, scaler):
        self.features = []
        self.targets = []
        if os.path.exists(f'{path}.csv'):
            print(f'Found cache for {path}.')
            with open(f'{path}.csv', 'r') as csvf:
                cache = csv.reader(csvf)
                for row in cache:
                    self.targets.append(int(row.pop()))
                    self.features.append(tuple([float(col) for col in row]))
            print(f'Loaded {path} from cache.')
        else:
            print(f'Parsing of {path} started.')
            with open(path, 'r') as txtf, open(f'{path}.csv', 'w') as csvf:
                extractor = FeatureExtractor()
                cache = csv.writer(csvf)
                text = []
                for line in txtf:
                    if line.startswith('---'):
                        self.targets.append(int(text.pop()))
                        extracted = extractor.extract(text[1:])
                        extracted = list(dataclasses.astuple(extracted))
                        self.features.append(extracted)
                        text = []
                        row = list(self.features[-1])
                        row.append(self.targets[-1])
                        cache.writerow(row)
                    else:
                        text.append(line)
            print(f'Parsing of {path} finished.')

        if 'train' in path:
            scaler.fit(self.features)
        self.features = scaler.transform(self.features)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        features = self.features[index]
        target = self.targets[index]
        return torch.tensor(features).float(), torch.tensor(target).float()


def tune_threshold(y, y_score):
    threshold = 0.0
    best_threshold = None
    best_score = 0.0

    while threshold <= 1.0:
        y_pred = y_score > threshold
        f2_score = fbeta_score(y, y_pred, beta=2)
        if f2_score >= best_score:
            best_threshold = threshold
            best_score = f2_score
        threshold += 0.01

    assert best_threshold is not None
    return best_threshold, best_score


def print_metrics(y, y_pred, y_score):
    auc_score = average_precision_score(y, y_score)
    f2_score = fbeta_score(y, y_pred, beta=2)
    print(f'[Test ] AUC score: {auc_score:.2f} F2 score: {f2_score:.2f}')
    print('[Test ] Classification Report:')
    print(classification_report(y, y_pred))
    print('[Test ] Confusion Matrix:')
    print(confusion_matrix(y, y_pred))


def train(model, loss_fn, loader, optimizer, epoch, log_interval):
    model.train()
    for batch, (X, y) in enumerate(loader):
        pred = model(X)
        loss = loss_fn(pred, y)
        loss = loss.mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % log_interval == 0:
            loss = loss.item()
            print(f"[Train] Epoch: {epoch} Progress: {batch}/{len(loader)} Loss: {loss:.5f}")


def validate(model, loss_fn, loader):
    model.eval()
    test_loss = 0
    y_true = torch.tensor([])
    y_score = torch.tensor([])

    with torch.no_grad():
        for X, y in loader:
            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.mean()
            y_true = torch.concat([y_true, y])
            y_score = torch.concat([y_score, torch.sigmoid(pred)])

    threshold, f2_score = tune_threshold(y_true, y_score)
    model.threshold = torch.tensor(threshold)
    avg_loss = test_loss / len(loader)

    print(f"[Valid] F2 score: {f2_score:.5f} Average loss: {avg_loss:.5f}")


def test(model, loader):
    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])

    with torch.no_grad():
        for X, y in loader:
            pred = model(X)
            y_true = torch.concat([y_true, y])
            y_score = torch.concat([y_score, torch.sigmoid(pred)])

    y_pred = y_score > model.threshold
    print_metrics(y_true, y_pred, y_score)


def test_ml(clf, train_dataset, valid_dataset, test_dataset):
    clf.fit(train_dataset.features, train_dataset.targets)
    X = valid_dataset.features
    y = valid_dataset.targets
    y_score = clf.predict_proba(X)[:, 1]
    threshold, _ = tune_threshold(y, y_score)

    X = test_dataset.features
    y = test_dataset.targets
    y_score = clf.predict_proba(X)[:, 1]
    y_pred = y_score > threshold
    print_metrics(y, y_pred, y_score)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', type=str, default=None, metavar='path',
                        required=False, help='training data file')
    parser.add_argument('--valid', type=str, default=None, metavar='path',
                        required=False, help='validation data file')
    parser.add_argument('--test', type=str, default=None, metavar='path',
                        required=False, help='validation data file')
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
                        help=' batch size for training (default: 64)')
    parser.add_argument('--eval-batch-size', type=int, default=1000, metavar='N',
                        help='batch size for evaluation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='for saving the current model')
    parser.add_argument('--ml', action='store_true', default=False,
                        help='train and evaluate some machine learning models')
    args = parser.parse_args()

    if args.ml:
        if not args.train:
            print('[Train] Error: train dataset is not presented',
                  file=sys.stderr)
            sys.exit(1)
        if not args.valid:
            print('[Valid] Error: validation dataset is not presented',
                  file=sys.stderr)
            sys.exit(1)
        if not args.test:
            print('[Test ] Error: test dataset is not presented',
                  file=sys.stderr)
            sys.exit(1)

        scaler = StandardScaler()
        train_dataset = SySeVRDataset(args.train, scaler)
        valid_dataset = SySeVRDataset(args.valid, scaler)
        test_dataset = SySeVRDataset(args.test, scaler)

        print('\n###############     Decision Tree     ###############')
        clf = DecisionTreeClassifier(random_state=args.seed)
        test_ml(clf, train_dataset, valid_dataset, test_dataset)

        print('\n###############     Random Forest     ###############')
        clf = RandomForestClassifier(random_state=args.seed)
        test_ml(clf, train_dataset, valid_dataset, test_dataset)
    else:
        if args.train:
            if not args.valid:
                print('[Valid] Error: validation dataset is not presented',
                    file=sys.stderr)
                sys.exit(1)

            scaler = StandardScaler()
            train_dataset = SySeVRDataset(args.train, scaler)
            valid_dataset = SySeVRDataset(args.valid, scaler)
            train_dataloader = DataLoader(train_dataset, args.train_batch_size)
            valid_dataloader = DataLoader(valid_dataset, args.eval_batch_size)

            mlp = MultilayerPerceptron(len(train_dataset[0][0]))
            if os.path.exists('mlp.pt'):
                mlp.load_state_dict(torch.load('mlp.pt'))
            optimizer = optim.SGD(mlp.parameters(), args.lr)
            loss = nn.BCEWithLogitsLoss()
            for epoch in range(1, args.epochs + 1):
                train(mlp, loss, train_dataloader, optimizer, epoch,
                    args.log_interval)
                validate(mlp, loss, valid_dataloader)

            torch.save(mlp.state_dict(), 'mlp.pt')
            with open('scaler.pt', 'wb') as f:
                pickle.dump(scaler, f)

    if args.test:
        print('\n############### Multilayer Perceptron ###############')
        with open('scaler.pt', 'rb') as f:
            scaler = pickle.load(f)
        test_dataset = SySeVRDataset(args.test, scaler)
        test_dataloader = DataLoader(test_dataset, args.eval_batch_size)
        mlp = MultilayerPerceptron(len(test_dataset[0][0]))
        mlp.load_state_dict(torch.load('mlp.pt'))
        print(mlp.threshold)
        test(mlp, test_dataloader)


if __name__ == '__main__':
    main()
