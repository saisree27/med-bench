import csv
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForQuestionAnswering
import torch
from datasets import load_dataset
from torch.nn.functional import cross_entropy, one_hot, softmax
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import time
from torcheval.metrics import MulticlassAccuracy
import matplotlib.pyplot as plt
import sacremoses
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from torch.nn import DataParallel
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

torch.cuda.init()
num_gpus = torch.cuda.device_count()
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 256 
TEST_BATCH_SIZE = 256
LR = 0.0001
OPTIMIZER = "adam"
EPOCHS = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

mednli_dataset = load_dataset('hippocrates/MedNLI_test')
mednli_train = mednli_dataset['train']
mednli_test= mednli_dataset['test']
mednli_valid = mednli_dataset['valid']

mednli_biogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt", use_fast=True)
mednli_biogpt_model = AutoModelForSequenceClassification.from_pretrained("microsoft/biogpt", num_labels=3, problem_type="multi_label_classification")

# mednli_clinical_bert_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
# mednli_clinical_bert_model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=3, problem_type="multi_label_classification")

# mednli_bio_bert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
# mednli_bio_bert_model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-v1.1", num_labels=3, problem_type="multi_label_classification")

if num_gpus > 1:
    mednli_biogpt_model = DataParallel(mednli_biogpt_model).to(device)
    # mednli_clinical_bert_model = DataParallel(mednli_clinical_bert_model).to(device)
    # mednli_bio_bert_model = DataParallel(mednli_bio_bert_model).to(device)

print(f"Using {device} with {num_gpus} GPUS.")

def change_y(y):
    for i in range(len(y)):
        if y[i] == "entailment":
            y[i] = 0
        elif y[i] == "contradiction":
            y[i] = 1
        else:
            y[i] = 2
    res = one_hot(torch.tensor(y), num_classes=3)
    return res


def loss_fn(y_hat, y):
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float64)
    else:
        y = y.to(dtype=torch.float64)
    return cross_entropy(y_hat, y)

def plot(model_name, t_loss, v_loss):
    if isinstance(t_loss[0], torch.Tensor): 
        t_loss = [x.cpu().item() if x.is_cuda else x.item() for x in t_loss]
    if isinstance(v_loss[0], torch.Tensor):
        v_loss = [x.cpu().item() if x.is_cuda else x.item() for x in v_loss]
    plt.plot(range(len(t_loss)), t_loss, label='Training Loss')
    plt.plot(range(len(t_loss)), v_loss, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(model_name + " Loss over Epochs")
    plt.legend()

    timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
    plt.savefig(model_name + "_" + str(timestamp) + "_loss.png")

def train(tokenizer, model):
    model = model.to(device)
    X_train = mednli_train["query"]
    y_train = change_y(mednli_train["answer"])

    X_test = mednli_test["query"]
    y_test = change_y(mednli_test["answer"])

    X_valid = mednli_valid["query"]
    y_valid = change_y(mednli_valid["answer"])

    y_train = y_train.to(device)
    y_valid = y_valid.to(device)
    y_test = y_test.to(device)

    X_train_token = tokenizer(X_train, return_tensors='pt', padding=True, truncation=False)
    train_dataset = TensorDataset(X_train_token['input_ids'], X_train_token['attention_mask'],  y_train)
    train_data_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)

    X_valid_token = tokenizer(X_valid, return_tensors='pt', padding=True, truncation=False)
    valid_dataset = TensorDataset(X_valid_token['input_ids'], X_valid_token['attention_mask'],  y_valid)
    valid_data_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=True, num_workers=0)

    optimizer = Adam(model.parameters(), lr=LR)

    tr_losses = []
    val_losses = []
    val_acc = []

    for epoch in range(EPOCHS):
        torch.cuda.empty_cache()
        model.train()
        training_loss = 0
        print("Train")
        for i, batch in enumerate(train_data_loader, 0):
            input_ids, attention_masks, y = batch
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = model(input_ids, attention_masks).logits
            L = loss_fn(output, y)
            training_loss += len(input_ids) * L
            L.backward()

            optimizer.step()
        print("Valid")
        model.eval()

        with torch.no_grad():
            metric = MulticlassAccuracy()
            total_valid_loss = 0
            for i, batch in enumerate(valid_data_loader, 0):
                input_ids, attention_masks, y = batch
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                y = y.to(device)

                valid_output = model(input_ids, attention_masks).logits
                L = loss_fn(valid_output, y)
                total_valid_loss += len(input_ids) * L
                
                _, labels = y.max(dim=1)
                metric.update(softmax(valid_output, dim=None), labels)
            

            val_losses.append(total_valid_loss)
            tr_losses.append(training_loss / len(X_train))
            val_acc = metric.compute().item()

            print('\tEPOCH: ', epoch, ', Training Loss: ', training_loss / len(X_train), ', Validation Loss: ', total_valid_loss / len(X_valid), ', Validation Acc: ', val_acc)

    timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
    torch.save(model.state_dict(), type(model).__name__ + "_" + timestamp + ".pth")

    plot(type(model).__name__, tr_losses, val_losses)

    torch.save(model.state_dict(), 'model.pt')

    testing_loss = 0
    metric = MulticlassAccuracy()

    torch.cuda.empty_cache()

    X_test_token = tokenizer(X_test, return_tensors='pt', padding=True, truncation=False)
    test_dataset = TensorDataset(X_test_token['input_ids'], X_test_token['attention_mask'],  y_test)
    test_data_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=0)

    print("Testing")
    with torch.no_grad():
        for i, batch in enumerate(test_data_loader, 0):
            input_ids, attention_masks, y = batch
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            y = y.to(device)

            output = model(input_ids, attention_masks).logits

            L = loss_fn(output, y)
            testing_loss += len(input_ids) * L
            
            _, labels = y.max(dim=1)
            metric.update(softmax(output, dim=None), labels)

    print("Test Loss: ", testing_loss.item() / len(X_test))
    print("Test Accuracy: ", metric.compute().item())

    return tr_losses, val_losses, val_acc

print("Training BioGPT Model")
tr_loss, val_loss, val_acc = train(mednli_biogpt_tokenizer, mednli_biogpt_model)
print(tr_loss, val_loss, val_acc)

# print("Training Clincal Bert Model")
# tr_loss, val_loss, val_acc = train(mednli_clinical_bert_tokenizer, mednli_clinical_bert_model)
# print(tr_loss, val_loss, val_acc)

# print("Training Bio Bert Model")
# tr_loss, val_loss, val_acc = train(mednli_bio_bert_tokenizer, mednli_bio_bert_model)
# print(tr_loss, val_loss, val_acc)