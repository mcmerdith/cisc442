# %% [markdown]
# # Part 2 - Custom Classification

# %%
from torchsummary import summary
from os import makedirs, path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU is not available")

dataset_root = "sample_data"
output_dir = path.join("output", "part-2")
makedirs(output_dir, exist_ok=True)

model_name = "model.pth"
# model = None

# %% [markdown]
# ### Custom VGG16 model

# %%
# Adapted from https://github.com/mcmerdith/cisc484/blob/hw4/HW%204_CNN-1.ipynb


class ModifiedVGG16ConvUnit(nn.Module):
    def __init__(self, input_channels, out_channels, rate=0.3, drop=True):
        super().__init__()
        # This line creates a 2D convolutional layer using PyTorch's nn.Conv2d module.
        self.conv = nn.Conv2d(input_channels, out_channels, 3, 1, 1)
        # self.bn = nn.BatchNorm2d(out_channels) # normalize the activations of the convolutional layer in the neural network.
        # introduces non-linearity (-ve --> zeros)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(rate)
        self.drop = drop

    # This method defines the forward pass of the convolutional block.
    def forward(self, x):
        # x = self.relu(self.bn(self.conv(x)))
        x = self.relu(self.conv(x))
        if self.drop:
            x = self.dropout(x)
        return x


def vgg16_layer(input_channels, out_channels, num, dropout=0.3, pool=True):
    layers = []
    for _ in range(num):
        layers.append(ModifiedVGG16ConvUnit(input_channels,
                      out_channels, dropout, drop=False))
        input_channels = out_channels
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return layers


class ModifiedVGG16(nn.Module):
    def __init__(self, n_classes: int):
        super(ModifiedVGG16, self).__init__()
        # Each layer should have 2-3 convolution blocks
        # but a single block is used to reduce training time.
        self.features = nn.Sequential(
            *vgg16_layer(3, 64, 1),     # 224x224 -> 112x112
            *vgg16_layer(64, 128, 1),   # 112x112 -> 56x56
            *vgg16_layer(128, 256, 1),  # 56x56 -> 28x28
            *vgg16_layer(256, 256, 1),  # 28x28 -> 14x14
            *vgg16_layer(256, 256, 1),  # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, n_classes),
            nn.Flatten(),
        )

    # defines how data flows through the network during the forward pass.
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# %% [markdown]
# ### Setup and shared functions

# %%


def save_model_state(model_state: dict):
    torch.save(model_state, path.join(output_dir, model_name))


def load_model_state():
    return torch.load(path.join(output_dir, model_name))


def make_model(n_classes=100):
    return ModifiedVGG16(n_classes).to(device)


# setup transformers
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

batch_size = 64


def train_dataset():
    trainset = datasets.CIFAR100(
        dataset_root, train=True, transform=data_transforms, download=True)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                               num_workers=2, pin_memory=True, persistent_workers=True)

    return trainset, train_loader


def test_dataset():
    testset = datasets.CIFAR100(
        dataset_root, train=False, transform=data_transforms, download=True)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True,
                                              num_workers=2, pin_memory=True, persistent_workers=True)

    return testset, test_loader

# %% [markdown]
# ### Training


# %%
trainset, train_loader = train_dataset()
testset, test_loader = test_dataset()
n_train, n_test = len(trainset), len(testset)

# build the model
model = make_model()

# training time
num_epochs = 30
criterion = nn.CrossEntropyLoss()
# higher learning rate to train in less epochs
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# save the best model
best_epoch = 0
best_acc = 0.0
best_weights = None


def test():
    test_loss = 0.0
    test_correct = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # score the model
            _, preds = torch.max(outputs, 1)
            test_loss += loss.item() * images.size(0)
            test_correct += torch.sum(preds == labels.data)

    return test_loss / n_test, test_correct / n_test


for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    train_loss = 0.0
    train_correct = 0.0
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # reset gradients
        optimizer.zero_grad()
        # calculate loss and backprop
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 20)

        optimizer.step()

        # score the model
        _, preds = torch.max(outputs, 1)
        train_loss += loss.item() * images.size(0)
        train_correct += torch.sum(preds == labels)

    # finished with this epoch
    scheduler.step()

    epoch_loss, epoch_acc = test()
    print(f"    train loss: {train_loss/n_train} acc: {train_correct/n_train}")
    print(f"    test loss: {epoch_loss} acc: {epoch_acc}")

    # store if this is the best iteration
    if epoch_acc > best_acc or not best_weights:
        best_epoch = epoch
        best_acc = epoch_acc
        best_weights = model.state_dict()

# force types
assert best_weights is not None, "no iterations were successful???"
save_model_state(best_weights)

print(f"Best iteration: {best_epoch + 1}, acc: {best_acc}")
print(f"Saved model to {model_name}")

# %% [markdown]
# ### Testing

# %%
# load the best model
if not model:
    # create a blank model if one doesn't exist
    model = make_model()

model.load_state_dict(load_model_state())

# %%
# load the test set
testset, test_loader = test_dataset()

correct = 0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        # score the model
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)


print(f"Test accuracy: {correct / len(testset)}")

# %% [markdown]
# ### Debugging

# %%
model = make_model()
summary(model, (3, 224, 224))
