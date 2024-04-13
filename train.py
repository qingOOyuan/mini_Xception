import os
import torch
import datetime
import numpy as np
from visualdl import LogWriter
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from utils.Model import mini_XCEPTION
from utils.dataset import FER2013

num_epochs = 200
log_interval = 100      # Interval for printing information
num_workers = 10        # Number of threads

# Output directory, named after the current time
output_folder = 'output/{}/'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H.%M.%S"))
writer = LogWriter(logdir=output_folder)

batch_size = 32
input_size = (48, 48)
num_classes = 7
patience = 50

# Create output directory if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = mini_XCEPTION(num_classes=num_classes)
model.to(device)

# Load data
train_dataset = FER2013("train", input_size=input_size)
test_dataset = FER2013("test", input_size=input_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Define the optimizer
optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())
loss_function = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='max',
                                                       factor=0.1,
                                                       patience=int(patience / 4),
                                                       verbose=True)

def train():
    """
    The function for training the model.
    """
    best_accuracy = 0
    global_step = 0
    for epoch in range(num_epochs):
        total_train_loss, total_test_loss = 0, 0
        total_train_accuracy, total_test_accuracy = 0, 0
        batch_count = 0
        final_batch_index = len(train_loader) - 1
        model.train()
        for batch_index, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            predicted_labels = model(images)
            # Compute accuracy and loss
            accuracy = accuracy_score(np.argmax(predicted_labels.cpu().detach().numpy(), axis=-1), labels)
            total_train_accuracy += accuracy
            loss = loss_function(predicted_labels, labels.to(device))
            total_train_loss += loss.item()
            batch_count += 1
            # Update gradients
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            mean_train_accuracy = total_train_accuracy / batch_count
            mean_train_loss = total_train_loss / batch_count

            global_step += 1
            writer.add_scalar(tag="train_accuracy", step=global_step, value=mean_train_accuracy)
            writer.add_scalar(tag="train_loss", step=global_step, value=mean_train_loss)

            if batch_index % log_interval == 0 or batch_index == final_batch_index:
                print(f"Epoch: {epoch}\tIteration: {batch_index}/{final_batch_index}\tLoss: {mean_train_loss:.4f}\tAccuracy: {mean_train_accuracy:.4f}")
        batch_count = 0
        model.eval()
        for batch_index, (images, labels) in enumerate(test_loader):
            predicted_labels = model(images.to(device))
            accuracy = accuracy_score(np.argmax(predicted_labels.cpu().detach().numpy(), axis=-1), labels)
            loss = loss_function(predicted_labels, labels.to(device))
            total_test_loss += loss.item()
            total_test_accuracy += accuracy
            batch_count += 1

        mean_test_loss = total_test_loss / batch_count
        mean_test_accuracy = total_test_accuracy / batch_count
        
        scheduler.step(mean_test_accuracy)
        print(f"Evaluation\tLoss: {mean_test_loss:.4f}\tAccuracy: {mean_test_accuracy:.4f}")

        writer.add_scalar(tag="test_accuracy", step=epoch, value=mean_test_accuracy)
        writer.add_scalar(tag="test_loss", step=epoch, value=mean_test_loss)

        if mean_test_accuracy > best_accuracy:
            model_save_path = f"{output_folder}/Epoch_{epoch}_Accuracy_{mean_test_accuracy:.4f}.pth"
            torch.save(model.state_dict(), model_save_path)
            best_accuracy = mean_test_accuracy
            print("Saved best model")

if __name__ == "__main__":
    train()
