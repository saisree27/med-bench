from numpy.random import RandomState
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
import torch
from torch.nn import DataParallel
from torch.nn.functional import cross_entropy, one_hot, softmax
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import time
from torcheval.metrics import MulticlassAccuracy
import matplotlib.pyplot as plt
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

print("DISEASE SYMPTOMS TRAIN")

asq_train = pd.read_csv('augmented_bioasq_dataset_train.csv')
asq_test = pd.read_csv('augmented_bioasq_dataset_test.csv')

# MODEL = "microsoft/biogpt"
# MODEL = "emilyalsentzer/Bio_ClinicalBERT"
MODEL = "dmis-lab/biobert-v1.1"

bioasq_tokenizer = AutoTokenizer.from_pretrained(MODEL)
bioasq_model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

BATCH_SIZE = 16
LR = 0.00001
OPTIMIZER = "adam"
EPOCHS = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


def loss_fn(y_hat, y):
    _, labels = y.max(dim=1)
    return cross_entropy(y_hat, labels)

def change_y(y):
    for i in range(len(y)):
        y[i] = 0 if y[i] == 'no' else 1

    labels = y.values.astype(np.int64)
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
    plt.savefig('../plots/' + model_name + "_" + str(timestamp) + "_bioasq_loss.png")

def train(tokenizer, model):
    print("TRAINING")
    model = DataParallel(model).to(device)

    X_train = asq_train[['question', 'context']].apply(tuple, axis=1)
    X_train = [x[0] + x[1] for x in X_train]
    y_train = change_y(asq_train["label"])

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    batch_idx = torch.arange(0, len(X_train), BATCH_SIZE)
    valid_batch_idx = torch.arange(0, len(X_valid), BATCH_SIZE)

    optimizer = Adam(model.parameters(), lr=LR)

    tr_losses = []
    val_losses = []
    val_acc = []

    for epoch in range(EPOCHS):
        torch.cuda.empty_cache()
        model.train()
        training_loss = 0
        for i in batch_idx:
            j = i.item()
            X = X_train[j: j + BATCH_SIZE]
            y = y_train[j: j + BATCH_SIZE].to(device)

            encoded_input = tokenizer(X, padding=True, truncation=True, add_special_tokens=True, max_length=500, return_tensors="pt").to(device)
            output = model(**encoded_input).logits
            del encoded_input
        
            L = loss_fn(output, y)
            training_loss += len(X) * L

            optimizer.zero_grad()

            L.backward()

            optimizer.step()

        model.eval()

        with torch.no_grad():
            metric = MulticlassAccuracy()
            valid_loss = 0
            for i in valid_batch_idx:
                j = i.item()
                X = X_valid[j: j + BATCH_SIZE]
                y = y_valid[j: j + BATCH_SIZE]
                encoded_input = tokenizer(X, padding=True, truncation=True, add_special_tokens=True, max_length=500, return_tensors="pt").to(device)
                valid_output = model(**encoded_input).logits.cpu()
                del encoded_input

                valid_loss += loss_fn(valid_output, y) * len(X)

                _, labels = y.max(dim=1)
                metric.update(softmax(valid_output), labels)
           
            tr_losses.append(training_loss.item() / len(X_train))
            val_losses.append(valid_loss.item() / len(X_valid))

            acc = metric.compute().item()
            val_acc.append(acc)

            print('\tEPOCH: ', epoch, ', Training Loss: ', training_loss.item() / len(X_train), ', Validation Loss: ', valid_loss.item() / len(X_valid), ', Validation Acc: ', acc)

    timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
    torch.save(model.state_dict(), type(model).__name__ + "_" + timestamp + "disease_symptoms.pth")

    plot(type(model).__name__, tr_losses, val_losses)

    torch.cuda.empty_cache()
    return tr_losses, val_losses, val_acc

def eval_on_test(checkpoint, model, tokenizer):
    X_test = asq_test[['question', 'context']].apply(tuple, axis=1)
    X_test = [x[0] + x[1] for x in X_test]
    y_test = change_y(asq_test["label"])

    model = DataParallel(model).to(device)
    model.load_state_dict(torch.load(checkpoint))

    test_batch_idx = torch.arange(0, len(X_test), BATCH_SIZE)
    testing_loss = 0
    metric = MulticlassAccuracy()
    with torch.no_grad():
        for i in test_batch_idx:
            j = i.item()
            X = X_test[j: j + BATCH_SIZE]
            y = y_test[j: j + BATCH_SIZE]

            encoded_input = tokenizer(X, padding=True, truncation=True, add_special_tokens=True, max_length=500, return_tensors="pt").to(device)
            output = model(**encoded_input).logits.cpu()

            del encoded_input

            L = loss_fn(output, y)
            testing_loss += len(X) * L

            _, labels = y.max(dim=1)
            metric.update(softmax(output), labels)

    print("Test Loss: ", testing_loss.item() / len(X_test))
    print("Test Accuracy: ", metric.compute().item())



# tr_loss, val_loss, val_acc = train(bioasq_tokenizer, bioasq_model)
# print(tr_loss, val_loss, val_acc)

checkpt = "BioBERTDataParallel_Apr-27-2024_2120disease_symptoms.pth"
eval_on_test(checkpt, bioasq_model, bioasq_tokenizer)