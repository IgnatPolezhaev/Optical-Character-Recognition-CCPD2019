import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import cv2 as cv

available_characters = '0123456789abcdefghijklmnopqrstuvwxyz'

# encoding text
def encoder_text(text):
    enc_text = np.zeros(len(text))
    text = text.lower()
    for i in range(len(text)):
        for j in range(len(available_characters)):
            if text[i] == available_characters[j]:
                enc_text[i] = j
    return enc_text.astype(np.float32)

#decoding text
def decoder_text(enc_text, prob=True):
    # enc_text = [b_s, 37, src_len]
    b_s = enc_text.shape[0]
    s_l = enc_text.shape[-1]
    batch_text = []
    for i in range(b_s):
        text = ''
        if prob:
            ind = torch.argmax(enc_text[i], dim=0)
        else:
            ind = enc_text[i]
        for j in range(s_l):
            text += available_characters[int(ind[j].item())]
        batch_text.append(text)
    return batch_text

# get data
def get_data(root):
    data = []
    dim = (256, 128)

    photos = os.listdir(root)

    for item in tqdm(photos):
        if item != '.DS_Store':
            tmp_data = []
            img = cv.imread(root+item)
            img_resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
            img_grey = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
            tmp_data.append(img_grey.astype(np.float32))
            text = item.split('-')[1].split('.')[0][1:]
            enc_text = encoder_text(text)
            tmp_data.append(enc_text)
            data.append(tmp_data)

    return data

def train(model, device, dataloader, optimizer, criterion, clip, train_history=None, valid_history=None):
    model.train()
    
    epoch_loss = 0
    history = []
    iter = 0

    for img, text in dataloader:
        img = img.to(device)
        text = text.to(device)
        # text = [b_s, src_len]
        #img = [b_s, 128, 256]
        optimizer.zero_grad()
        output = model(img)
        #output = [b_s, src_len, output_size]
        loss = criterion(output, text.long())
        loss.backward()
        
        # Let's clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
        history.append(loss.cpu().data.numpy())

        if (iter+1)%10==0:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

            clear_output(True)
            ax[0].plot(history, label='train loss')
            ax[0].set_xlabel('Batch')
            ax[0].set_title('Train loss')
            if train_history is not None:
                ax[1].plot(train_history, label='general train history')
                ax[1].set_xlabel('Epoch')
            if valid_history is not None:
                ax[1].plot(valid_history, label='general valid history')
            plt.legend()
            
            plt.show()

        iter += 1

        
    return epoch_loss / len(dataloader)

def evaluate(model, device, dataloader, criterion):
    
    model.eval()
    epoch_loss = 0
    history = []
    
    with torch.no_grad():
        for img, text in dataloader:
            img = img.to(device)
            text = text.to(device)

            output = model(img)
            loss = criterion(output, text.long())
            
            epoch_loss += loss.item()
            history.append(loss.cpu().data.numpy())
        
    return epoch_loss / len(dataloader)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def CER_metric(pred, true):
    # pred = [b_s, 37, src_len]
    pred = torch.argmax(pred, dim=1)
    # pred = [b_s, src_len]
    # true = [b_s, src_len]
    b_s = pred.shape[0]
    s_l = pred.shape[1]
    cer_met = torch.count_nonzero(pred==true)/b_s/s_l
    return cer_met

def accuracy(pred, true):
    # pred = [b_s, 37, src_len]
    pred = torch.argmax(pred, dim=1)
    # pred = [b_s, src_len]
    # true = [b_s, src_len]
    b_s = pred.shape[0]
    text_pred = decoder_text(pred, prob=False)
    text_true = decoder_text(true, prob=False)
    acc = 0
    for i in range(b_s):
        if text_pred[i] == text_true[i]:
            acc += 1
    return acc/b_s