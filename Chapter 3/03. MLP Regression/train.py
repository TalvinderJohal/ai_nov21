#import the needed libraries

# yes, you can import your code. Cool!
from data_handler import load_data, to_batches
import torch
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

from model import MLP

url = "/Users/talvinderjohal/Desktop/Talvinder Strive Course/ai_nov21/Chapter 3/03. MLP Regression/data/turkish_stocks.csv"
x_train, x_test, y_train, y_test = load_data(url)
print(x_train.shape)


mlp = MLP()

optimizer = optim.SGD(mlp.parameters(), lr=2e-5)
loss_func = nn.MSELoss()

epochs = 1000
loss_list = []

for i in range(epochs):
    running_loss = 0
    print(f"Epoch:  {i}")
    x_train_batch, x_test_batch, y_train_batch, y_test_batch = to_batches(x_train, x_test, y_train, y_test, 10)

    for x_batch, y_batch in zip(x_train_batch, y_train_batch):
        
        optimizer.zero_grad()

        prediction = mlp.forward(x_batch)
        loss = loss_func(prediction, y_batch) 
        
        loss.backward()        
        optimizer.step()   

        running_loss += loss.item()
    
    print(f"loss: {running_loss/x_train_batch.shape[0]}")
    loss_list.append(running_loss/x_train_batch.shape[0])

plt.plot(loss_list)
plt.show()

# Remember to validate your model: with torch.no_grad() ...... model.eval .........model.train
