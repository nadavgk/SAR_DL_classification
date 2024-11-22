import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py


# Custom Dataset Class for Loading Data from .h5 Files
# this class cals upon the data drom the h5 files
class SatelliteDataset(Dataset):
    def __init__(self, h5_file_path):
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.sen1_data = self.h5_file['sen1']
        self.sen2_data = self.h5_file['sen2']
        self.labels = self.h5_file['label'][:, :]  # Load the full dataset into memory

    def __len__(self):
        return self.labels.shape[0]  # Correctly returns the total number of samples

    def __getitem__(self, idx):
        sen1_image = self.sen1_data[idx]  # Shape (32, 32, 8)
        sen2_image = self.sen2_data[idx]  # Shape (32, 32, 10)
        label = self.labels[idx]

        # Convert to PyTorch tensors
        sen1_image = torch.tensor(sen1_image, dtype=torch.float32).permute(2, 0, 1)  # (8, 32, 32)
        sen2_image = torch.tensor(sen2_image, dtype=torch.float32).permute(2, 0, 1)  # (10, 32, 32)

        # Convert one-hot encoded label to class index
        label = torch.tensor(label, dtype=torch.float32)
        label = torch.argmax(label).long()  # Converts one-hot vector to class index

        return sen1_image, sen2_image, label

# CNN Model Class
# this is the architecture of the cnn model
class ConvNet(nn.Module):
    def __init__(self, num_classes=17):
        super(ConvNet, self).__init__()

        # Sentinel-1 branch
        self.sen1_conv1 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.sen1_dropout1 = nn.Dropout(p=0.25)
        self.sen1_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.sen1_dropout2 = nn.Dropout(p=0.25)
        self.sen1_pool = nn.MaxPool2d(2, 2)

        # Sentinel-2 branch
        self.sen2_conv1 = nn.Conv2d(10, 32, kernel_size=3, padding=1)
        self.sen2_dropout1 = nn.Dropout(p=0.25)
        self.sen2_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.sen2_dropout2 = nn.Dropout(p=0.25)
        self.sen2_pool = nn.MaxPool2d(2, 2)

        # Fully connected layers after concatenation
        self.fc1 = nn.Linear(64 * 8 * 8 * 2, 128)  # *2 for concatenated features
        self.fc1_dropout = nn.Dropout(p=0.5)  # Dropout after fc1
        self.fc2 = nn.Linear(128, 64)
        self.fc2_dropout = nn.Dropout(p=0.5)  # Dropout after fc2
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, sen1, sen2):
        # Sentinel-1 forward pass
        x1 = F.relu(self.sen1_conv1(sen1))
        x1 = self.sen1_dropout1(x1)
        x1 = self.sen1_pool(F.relu(self.sen1_conv2(x1)))
        x1 = self.sen1_dropout2(x1)
        x1 = x1.view(x1.size(0), -1)  # Flatten

        # Sentinel-2 forward pass
        x2 = F.relu(self.sen2_conv1(sen2))
        x2 = self.sen2_dropout1(x2)
        x2 = self.sen2_pool(F.relu(self.sen2_conv2(x2)))
        x2 = self.sen2_dropout2(x2)
        x2 = x2.view(x2.size(0), -1)  # Flatten

        # Concatenate both branches
        x = torch.cat((x1, x2), dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc1_dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_dropout(x)
        x = self.fc3(x)  # Output layer with num_classes outputs

        return x


def get_data_loader(h5_file_path, batch_size=32, shuffle=True):
    dataset = SatelliteDataset(h5_file_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (sen1, sen2, labels) in enumerate(train_loader):
            sen1, sen2, labels = sen1.to(device), sen2.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(sen1, sen2)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

    print('Training complete')


def evaluate_model(model, test_loader, criterion, device='cuda'):
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for sen1, sen2, labels in test_loader:
            sen1, sen2, labels = sen1.to(device), sen2.to(device), labels.to(device)
            outputs = model(sen1, sen2)

            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Calculate accuracy
            predicted = torch.argmax(outputs, dim=1)
            true_labels = torch.argmax(labels, dim=1)
            correct_predictions += (predicted == true_labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples
    print(f'Average Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy

# Loaders for Training and Testing
train_loader = get_data_loader(r"F:\DS_project\m1613658\random\training_1_perc_subset.h5", batch_size=64, shuffle=True)
test_loader = get_data_loader(r"F:\DS_project\m1613658\random\testing_1_perc_subset.h5", batch_size=64, shuffle=False)

# Model, Loss, and Optimizer
model = ConvNet(num_classes=17)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda')

# Evaluate the Model
evaluate_model(model, test_loader, criterion, device='cuda')

# dataset_path = r"F:\DS_project\m1613658\random\training_1_perc_subset.h5"
#
# cls = SatelliteDataset(dataset_path)
#
# sen1_image, sen2_image, label = cls.__getitem__(0)
#
# data_loader = get_data_loader(dataset_path)
#
# # Iterate through the DataLoader
# counter = 0
# # Iterate through the DataLoader
# for i, (sen1_image, sen2_image, label) in enumerate(data_loader):
#     print(f"Batch {i}:")
#     print(f"Sen1 image shape: {sen1_image.shape}")  # Should be (batch_size, 8, 32, 32)
#     print(f"Sen2 image shape: {sen2_image.shape}")  # Should be (batch_size, 10, 32, 32)
#     print(f"Labels shape: {label.shape}")          # Should be (batch_size,)
#     counter += 1
#     if counter == 102:  # Break after 2 iterations
#         break

