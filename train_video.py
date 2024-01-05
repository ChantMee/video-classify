import logging
import os
import torch
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import VideoFrameDataset  # Custom dataset for video frames.
from models import r3d_18, r2plus1d_18  # Custom models (3D ResNet and R(2+1)D).
import warnings
warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output.

# Function for training one epoch.
def train_epoch(model, criterion, optimizer, dataloader, device, epoch, logger, writer):
    model.train()  # Set the model to training mode.
    losses = []
    all_label = []
    all_pred = []

    # print("train in device:" + str(device))

    for data in tqdm(dataloader):  # Progress bar for batches.
        inputs, labels = data['data'].to(device), data['label'].to(device)
        optimizer.zero_grad()  # Zero the gradients.

        # Forward pass.
        outputs = model(inputs)
        if isinstance(outputs, list):
            outputs = outputs[0]

        # Compute loss.
        loss = criterion(outputs, labels.squeeze())
        losses.append(loss.item())

        # Compute accuracy.
        prediction = torch.max(outputs, 1)[1]
        all_label.extend(labels.squeeze())
        all_pred.extend(prediction)
        score = accuracy_score(labels.squeeze().cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())

        # Backward pass and optimize.
        loss.backward()
        optimizer.step()

    # Compute the average loss & accuracy for the epoch.
    training_loss = sum(losses) / len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    training_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())

    # Logging to TensorBoard and logger.
    writer.add_scalars('Loss', {'train': training_loss}, epoch + 1)
    writer.add_scalars('Accuracy', {'train': training_acc}, epoch + 1)
    logger.info("Average Training Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch + 1, training_loss, training_acc * 100))

# Function for validating one epoch.
def val_epoch(model, criterion, dataloader, device, epoch, logger, writer):
    model.eval()  # Set the model to evaluation mode.
    losses = []
    all_label = []
    all_pred = []

    with torch.no_grad():  # Disable gradient calculation.
        for batch_idx, data in enumerate(dataloader):
            inputs, labels = data['data'].to(device), data['label'].to(device)
            outputs = model(inputs)
            if isinstance(outputs, list):
                outputs = outputs[0]
            loss = criterion(outputs, labels.squeeze())
            losses.append(loss.item())
            prediction = torch.max(outputs, 1)[1]
            all_label.extend(labels.squeeze())
            all_pred.extend(prediction)

    # Compute the average loss & accuracy for the epoch.
    validation_loss = sum(losses) / len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    validation_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())

    # Logging to TensorBoard and logger.
    writer.add_scalars('Loss', {'validation': validation_loss}, epoch + 1)
    writer.add_scalars('Accuracy', {'validation': validation_acc}, epoch + 1)
    logger.info("Average Validation Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch + 1, validation_loss, validation_acc * 100))

    return validation_acc

# Setting paths for data and logs.

sum_path = "r3d"
log_path = "logs/r3d.log"

# Setting up logging to file and TensorBoard.
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger('SLR')
writer = SummaryWriter(sum_path)

# GPU configuration.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters.
num_classes = 2
epochs = 36
batch_size = 16
learning_rate = 1e-5
sample_size = 128
sample_duration = 16  # Frame sampling duration.

torch.manual_seed(2023)

if __name__ == '__main__':
    # Data loading and preprocessing.
    transform = transforms.Compose([
        transforms.Resize([sample_size, sample_size]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # crop (fill with white), rotate, flip, normalize, add noise, to tensor
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop([sample_size, sample_size], padding=4, padding_mode='reflect'),
        transforms.Resize([sample_size, sample_size]),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([sample_size, sample_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = VideoFrameDataset(root_dir="data/train", frame_count=sample_duration, transform=transform_train)
    test_set = VideoFrameDataset(root_dir="data/test", frame_count=sample_duration, transform=transform_test)
    logger.info("Dataset samples: {}".format(len(train_set) + len(test_set)))
    logger.info("Training samples: {}".format(len(train_set)))
    logger.info("Testing samples: {}".format(len(test_set)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

    # Model selection and preparation.
    model = r3d_18(pretrained=True, num_classes=num_classes).to(device)
    # calculate the number of trainable parameters in the model
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total number of trainable parameters: {}".format(pytorch_total_params))

    # Loss criterion and optimizer setup.
    num_items_class = train_set.get_num_item_each_class()
    tot_num = sum(num_items_class)
    weight = [tot_num / num_items_class[i] for i in range(len(num_items_class))]
    weight = torch.FloatTensor(weight).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Training and validation loop.
    logger.info("Training Started".center(60, '#'))
    best_acc = 0.0
    for epoch in range(epochs):
        train_epoch(model, criterion, optimizer, train_loader, device, epoch, logger, writer)
        validation_acc = val_epoch(model, criterion, test_loader, device, epoch, logger, writer)
        # Save the model if it has the best accuracy so far.
        if best_acc < validation_acc:
            best_acc = validation_acc
            torch.save(model.state_dict(), "r3d.pth")
        logger.info("Epoch {} Model Saved".format(epoch + 1).center(60, '#'))

    logger.info("Training Finished".center(60, '#'))
