import os
import torch
from tqdm import tqdm
from models import CNN, RNN, ATTN, GNN
import sklearn.metrics
import numpy as np
from dataset import load_6_structure_data, load_100_structure_data, CopolymerData

# hyperparameters
model_name = 'RNN'
input_size = 6
hidden_size = 32
output_size = 1
num_layers = 3

batch_size = 4
num_epochs = 100
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
x_train, x_test, y_train, y_test = load_6_structure_data()
train_dataset = CopolymerData(x_train, y_train)
test_dataset = CopolymerData(x_test, y_test)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# build model
if model_name == 'CNN':
    model = CNN(input_size, hidden_size, output_size).to(device)
elif model_name == 'RNN':
    model = RNN(input_size, hidden_size, output_size, num_layers).to(device)
elif model_name == 'ATTN':
    model = ATTN(input_size, hidden_size, output_size).to(device)
elif model_name == 'GNN':
    model = GNN(input_size, hidden_size, output_size).to(device)
else:
    raise ValueError('model name should be CNN, RNN or ATTN')

# loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train
best_loss = 1e10
for epoch in range(num_epochs):
    model.train()
    pbar = tqdm(train_loader)
    pbar.set_description('Epoch {}'.format(epoch+1))
    avg_loss = 0
    for x, y in pbar:
        x = x.float().to(device)
        y = y.float().to(device)

        # forward
        outputs = model(x).squeeze(1)
        loss = criterion(outputs, y)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())
        avg_loss += loss.item()

    avg_loss /= len(train_loader)
    if best_loss > avg_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join('ckpt', model_name + '.pth'))
        print('Model Saved!')
    

    # test
    model.eval()
    with torch.no_grad():
        loss = 0
        predictions = []
        for x, y in test_loader:
            x = x.float().to(device)
            y = y.float().to(device)
            outputs = model(x).squeeze(1)
            loss += criterion(outputs, y).item()
            #
            predictions.append(outputs.cpu().numpy())
            #

        print('Epoch [{}/{}], Test Loss: {:.4f}'.format(epoch+1, num_epochs, loss/len(test_loader)))
        #下面是自己加的

        predictions = np.concatenate(predictions)
        true_values = y_test
        r2_score = sklearn.metrics.r2_score(true_values, predictions)
        print('Epoch [{}/{}], Test Loss: {:.4f}, R² Score: {:.4f}'.format(epoch + 1, num_epochs, loss / len(test_loader),r2_score))