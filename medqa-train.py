from numpy.random import RandomState
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
import torch
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
import sacremoses
from huggingface_hub import login
from torch.nn.functional import cross_entropy, one_hot, softmax
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import time
import matplotlib.pyplot as plt
from torcheval.metrics import MulticlassAccuracy
# login()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

num_gpus = torch.cuda.device_count()
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
LR = 0.00001
OPTIMIZER = "adam"
EPOCHS = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

medQA_dataset = load_dataset('bigbio/med_qa', trust_remote_code=True)
medQA_dataset_train = medQA_dataset['train']
medQA_dataset_test = medQA_dataset['test']
medQA_dataset_valid = medQA_dataset['validation']

medQA_biogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt", use_fast=True)
medQA_biogpt_model = AutoModelForSequenceClassification.from_pretrained("microsoft/biogpt", num_labels=5, problem_type="multi_label_classification")

# medQA_clinical_bert_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
# medQA_clinical_bert_model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=5, problem_type="multi_label_classification")

# medQA_bio_bert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
# medQA_bio_bert_model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-v1.1", num_labels=5, problem_type="multi_label_classification")

def appendAnswer(datapoint):
    mergedQA = datapoint["question"] + " Return only the letter. " + ', '.join(f"{item['key']} - {item['value']}" for item in datapoint["options"])
    return {"questionAndAnswers": mergedQA}

medQA_dataset_train = medQA_dataset_train.map(appendAnswer)
medQA_dataset_test = medQA_dataset_test.map(appendAnswer)
medQA_dataset_valid = medQA_dataset_valid.map(appendAnswer)

def loss_fn(y_hat, y):
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float64)
    else:
        y = y.to(dtype=torch.float64)
    return cross_entropy(y_hat, y)

def change_y(y):
    mapping = {"A" : 0, "B" : 1, "C" : 2, "D" : 3, "E" : 4}
    for i in range(len(y)):
        y[i] = mapping[y[i]]
    res = one_hot(torch.tensor(y), num_classes=5)
    return res


def plot(model_name, t_loss, v_loss):
    plt.plot(range(len(t_loss)), t_loss, label='Training Loss')
    plt.plot(range(len(t_loss)), v_loss, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("BioGPT" + " Loss over Epochs")
    plt.legend()

    timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
    plt.savefig(model_name + "_" + str(timestamp) + "_biogpt_loss.png")


def train(model, tokenizer):
    model = model.to(device)
    model = nn.DataParallel(model)

    X_train = medQA_dataset_train["questionAndAnswers"]
    y_train = change_y(medQA_dataset_train["answer_idx"])

    X_valid = medQA_dataset_valid["questionAndAnswers"]
    y_valid = change_y(medQA_dataset_valid["answer_idx"])

    X_test = medQA_dataset_test["questionAndAnswers"]
    y_test = change_y(medQA_dataset_test["answer_idx"])

    y_train = y_train.to(device)
    y_valid = y_valid.to(device)
    y_test = y_test.to(device)

    X_train_token = tokenizer(X_train, return_tensors='pt', padding=True, truncation=True, max_length=512)
    train_dataset = TensorDataset(X_train_token['input_ids'], X_train_token['attention_mask'],  y_train)
    train_data_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)

    X_valid_token = tokenizer(X_valid, return_tensors='pt', padding=True, truncation=True, max_length=512)
    valid_dataset = TensorDataset(X_valid_token['input_ids'], X_valid_token['attention_mask'],  y_valid)
    valid_data_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=True, num_workers=0)

    # batch_idx = torch.arange(0, len(X_train), TRAIN_BATCH_SIZE)
    # batch_idx_valid = torch.arange(0, len(X_valid), VALID_BATCH_SIZE)
    # print(batch_idx)

    optimizer = Adam(model.parameters(), lr=LR)

    tr_losses = []
    val_losses = []
    val_acc = []

    for epoch in range(EPOCHS):
        torch.cuda.empty_cache()
        model.train()
        training_loss = 0
        print("TRAIN")
        for i, batch in enumerate(train_data_loader, 0):
            print(i)
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

        print("VALID")
        model.eval()
        torch.cuda.empty_cache()
        metric = MulticlassAccuracy()
        with torch.no_grad():
            total_valid_loss = 0
            for i, batch in enumerate(valid_data_loader, 0):
                print(i)
                input_ids, attention_masks, y = batch
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                y = y.to(device)

                valid_output = model(input_ids, attention_masks).logits
                L = loss_fn(valid_output, y)
                total_valid_loss += len(input_ids) * L
                
                _, labels = y.max(dim=1)
                metric.update(softmax(valid_output), labels)

            val_losses.append(total_valid_loss.item() / len(X_valid))
            tr_losses.append(training_loss.item() / len(X_train))
            acc = metric.compute().item()
            val_acc.append(acc)

            print('\tEPOCH: ', epoch, ', Training Loss: ', training_loss.item() / len(X_train), ', Validation Loss: ', total_valid_loss.item() / len(X_valid), ', Validation Acc: ', acc)

    timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
    torch.save(model.state_dict(), type(model).__name__ + "_" + timestamp + ".pth")

    plot(type(model).__name__, tr_losses, val_losses)

    torch.cuda.empty_cache()
    return tr_losses, val_losses, val_acc

# print(disease_train[:6])
print(type(medQA_biogpt_model).__name__)
tr_loss, val_loss, val_acc = train(medQA_biogpt_model, medQA_biogpt_tokenizer)
print(tr_loss, val_loss, val_acc)


def eval_on_test(checkpoint, model, tokenizer):
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(checkpoint))

    testing_loss = 0
    metric = MulticlassAccuracy()

    torch.cuda.empty_cache()

    X_test = medQA_dataset_test["questionAndAnswers"]
    y_test = change_y(medQA_dataset_test["answer_idx"])

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
            metric.update(softmax(output, dim=1), labels)

    print("Test Loss: ", testing_loss.item() / len(X_test))
    print("Test Accuracy: ", metric.compute().item())

checkpt = "DataParallel_Apr-28-2024_1218.pth"
# eval_on_test(checkpt, medQA_clinical_bert_model, medQA_clinical_bert_tokenizer)





