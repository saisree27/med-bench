from numpy.random import RandomState
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
import torch
from torch.nn.functional import cross_entropy, one_hot, softmax
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import time
from torcheval.metrics import MulticlassAccuracy
import matplotlib.pyplot as plt



disease_train = pd.read_csv('augmented_disease_symptoms_train.csv')
disease_test = pd.read_csv('augmented_disease_symptoms_test.csv')

disease_biogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
disease_biogpt_model = AutoModelForSequenceClassification.from_pretrained("microsoft/biogpt", num_labels=41, problem_type="multi_label_classification")

BATCH_SIZE = 64
LR = 0.0001
OPTIMIZER = "adam"
EPOCHS = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def loss_fn(y_hat, y):
    _, labels = y.max(dim=1)
    return cross_entropy(y_hat, labels)

def change_y_train(y_train):
    mapping = {}
    prevMapping = 0
    for label in y_train:
        if label not in mapping:
            mapping[label] = prevMapping
            prevMapping += 1

    for i in range(len(y_train)):
        y_train[i] = mapping[y_train[i]]

    labels = y_train.values.astype(np.int64)
    res = one_hot(torch.from_numpy(labels), num_classes=41)
    return res, mapping

def change_y_test(y_test, mapping):
    for i in range(len(y_test)):
        y_test[i] = mapping[y_test[i]]

    labels = y_test.values.astype(np.int64)
    res = one_hot(torch.from_numpy(labels), num_classes=41)
    return res

def plot(model_name, t_loss, v_loss):
    plt.plot(range(len(t_loss)), t_loss, label='Training Loss')
    plt.plot(range(len(t_loss)), v_loss, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(model_name + " Loss over Epochs")
    plt.legend()

    timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
    plt.savefig('./plots/' + model_name + "_" + str(timestamp) + "_loss.png")

def train(tokenizer, model):
    model = model.to(device)

    X_train = disease_train["sentence"]
    y_train, mapping = change_y_train(disease_train["prognosis"])

    X_test = disease_test["sentence"]

    y_test = change_y_test(disease_test["prognosis"], mapping)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    X_train = X_train.values.tolist()
    X_valid = X_valid.values.tolist()
    X_test = X_test.values.tolist()

    y_train = y_train.to(device)
    y_valid = y_valid.to(device)
    y_test = y_test.to(device)

    batch_idx = torch.arange(0, len(X_train), BATCH_SIZE)

    optimizer = Adam(model.parameters(), lr=LR)

    tr_losses = []
    val_losses = []
    val_acc = []

    for epoch in range(EPOCHS):
        model.train()
        training_loss = 0
        for i in batch_idx:
            j = i.item()
            X = X_train[j: j + BATCH_SIZE]
            y = y_train[j: j + BATCH_SIZE]

            encoded_input = tokenizer(X, return_tensors='pt', padding=True, truncation=True).to(device)
            output = model(**encoded_input).logits
        
            L = loss_fn(output, y)
            training_loss += len(X) * L

            optimizer.zero_grad()

            L.backward()

            optimizer.step()

        model.eval()

        with torch.no_grad():
            encoded_input = tokenizer(X_valid, return_tensors='pt', padding=True, truncation=True).to(device)
            valid_output = model(**encoded_input).logits

            valid_loss = loss_fn(valid_output, y_valid)
            val_losses.append(valid_loss.item())

            metric = MulticlassAccuracy()
            _, labels = y_valid.max(dim=1)
            metric.update(softmax(valid_output), labels)
            acc = metric.compute().item()

            tr_losses.append(training_loss.item() / len(X_train))
            val_acc.append(acc)

            print('\tEPOCH: ', epoch, ', Training Loss: ', training_loss.item() / len(X_train), ', Validation Loss: ', valid_loss.item(), ', Validation Acc: ', acc)

    timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
    torch.save(model.state_dict(), type(model).__name__ + "_" + timestamp + ".pth")

    plot(type(model).__name__, tr_losses, val_losses)

    # Eval on test
    test_batch_idx = torch.arange(0, len(X_test), BATCH_SIZE)
    testing_loss = 0
    metric = MulticlassAccuracy()
    for i in test_batch_idx:
        j = i.item()
        X = X_test[j: j + BATCH_SIZE]
        y = y_test[j: j + BATCH_SIZE]

        encoded_input = tokenizer(X, return_tensors='pt', padding=True, truncation=True).to(device)
        output = model(**encoded_input).logits

        L = loss_fn(output, y)
        testing_loss += len(X) * L

        _, labels = y.max(dim=1)
        metric.update(softmax(valid_output), labels)

    print("Test Loss: ", testing_loss.item() / len(X_test))
    print("Test Accuracy: ", metric.compute().item())

    return tr_losses, val_losses, val_acc


tr_loss, val_loss, val_acc = train(disease_biogpt_tokenizer, disease_biogpt_model)
print(tr_loss, val_loss, val_acc)