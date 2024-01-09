import re
from matplotlib import pyplot as plt

log_file_path = r"C:\Users\chant\OneDrive\Courses\WOA7015 Advanced Machine Learning\final\新建文件夹\logs\r3d.log"

# Patterns for extracting losses and accuracies
loss_pattern = r'Average Training Loss of Epoch \d+: ([0-9.]+) \| Acc: [0-9.]+%\nAverage Testing Loss of Epoch \d+: ([0-9.]+) \| Acc: [0-9.]+%'
acc_pattern = r'Average Training Loss of Epoch \d+: [0-9.]+ \| Acc: ([0-9.]+)%\nAverage Testing Loss of Epoch \d+: [0-9.]+ \| Acc: ([0-9.]+)%'

# Read the log file and extract the required information
with open(log_file_path, 'r') as file:
    logs = file.read()
    losses = re.findall(loss_pattern, logs)
    accuracies = re.findall(acc_pattern, logs)

# Separating training and testing losses and accuracies
training_losses = [float(loss[0]) for loss in losses]
testing_losses = [float(loss[1]) for loss in losses]
training_accuracies = [float(acc[0]) for acc in accuracies]
testing_accuracies = [float(acc[1]) for acc in accuracies]

# Plotting
epochs = range(1, len(training_losses)+1)

# Plotting Losses and Accuracies
plt.figure(figsize=(20, 5))

# Subplot for Losses
plt.subplot(1, 2, 1)
plt.plot(epochs, training_losses, label='Training Loss')
plt.plot(epochs, testing_losses, label='Testing Loss')
plt.title('Training and Testing Losses Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Subplot for Accuracies
plt.subplot(1, 2, 2)
plt.plot(epochs, training_accuracies, label='Training Accuracy')
plt.plot(epochs, testing_accuracies, label='Testing Accuracy')
plt.title('Training and Testing Accuracies Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.show()