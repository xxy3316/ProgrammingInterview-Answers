import torch
from tqdm import tqdm
from MIL_model import model
from readData import train_dataloader,test_dataloader
import numpy as np
from matplotlib import pyplot as plt
from test import test

# define loss criterion
criterion = torch.nn.L1Loss().cuda()

num_epochs = 40

train_loss_ls = []

model = model().cuda()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005)

for epoch in range(num_epochs):
    print('############## EPOCH - {} ##############'.format(epoch +1))
    training_loss = 0
    validation_loss = 0
    if epoch == 15:
        print('hello')

    # train for one epoch
    print('******** training ********')

    num_predictions = 0

    pbar = tqdm(total=len(train_dataloader))

    model.train()
    for images, targets in train_dataloader:
        images = images.to(device)
        targets = targets.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        if len(images) != 100:
            continue

        # forward + backward + optimize
        y_logits = model(images)
        # targets = targets.float32()
        targets = float(sum(targets)) / float(len(targets))

        # loss = criterion(y_logits, targets)
        loss = abs(y_logits-targets)
        loss.backward()
        optimizer.step()

        # training_loss += loss.item( ) *targets.size(0)
        training_loss += loss.item()

        num_predictions += 100

        pbar.update(1)

    # training_loss /= num_predictions
    train_loss_ls.append(training_loss)
    print('Loss：',training_loss)
    if epoch == 30:
        torch.save(model.state_dict(), 'MIL_model.pth')
        print('Model had been saved')

    pbar.close()
x = np.arange(1,len(train_loss_ls)+1)
y =  2  * x +  5
plt.title("Loss curves")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.plot(x,train_loss_ls)
plt.show()
test(model)

