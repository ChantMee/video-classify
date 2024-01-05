import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.io import read_image
from torchvision.transforms import Resize, Compose
from torch.utils.data import Dataset
import random
torch.manual_seed(2023)
# Define the VideoFrameDataset class, which extends PyTorch's Dataset class.
class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, frame_count=16, transform=None):
        """
        Args:
            root_dir (string): Path to the dataset directory containing class folders.
            frame_count (int): Number of frames to extract from each video.
            transform (callable, optional): A function/transform to process the images.
        """
        self.root_dir = root_dir
        self.frame_count = frame_count
        self.transform = transform
        self.samples = []
        self.labels = []

        # Traverse through the directory to collect paths of all video frames and corresponding labels.
        for label, class_folder_name in enumerate(os.listdir(root_dir)):
            class_folder_path = os.path.join(root_dir, class_folder_name)
            for video_folder_name in os.listdir(class_folder_path):
                video_folder_path = os.path.join(class_folder_path, video_folder_name)
                frames = sorted(os.listdir(video_folder_path))
                if len(frames) < frame_count:
                    # Repeat some frames if there aren't enough.
                    frames = frames * (frame_count // len(frames)) + frames[:frame_count % len(frames)]
                self.samples.append([os.path.join(video_folder_path, frame) for frame in frames])
                self.labels.append(label)

    def __len__(self):
        # Return the total number of samples.
        return len(self.samples)

    def __getitem__(self, idx):

        # Retrieve and process frames for a given index.
        frame_paths = self.samples[idx]
        start_frame = 0
        if len(frame_paths) > self.frame_count:
            start_frame = random.randint(0, len(frame_paths) - self.frame_count)
        end_frame = len(frame_paths) - 1
        if end_frame - start_frame > self.frame_count:
            end_frame = random.randint(start_frame + self.frame_count, end_frame)
        indices = np.linspace(start_frame, end_frame, self.frame_count).astype(int)
        selected_frames = [frame_paths[i] for i in indices]

        images = []
        for frame in selected_frames:
            image = read_image(frame)  # Read the image.
            image = image.float()
            if self.transform:
                image = self.transform(image)  # Apply transformations.
            images.append(image)

        # Stack images into a tensor.
        images = torch.stack(images, dim=1)  # Result size: [3, 16, H, W]
        label = torch.tensor(self.labels[idx])

        return {'data': images, 'label': label}


def show_images(images, cols=4, title=""):
    """Display a list of images in a grid, with frame number."""
    n_images = len(images)
    rows = n_images // cols + int(n_images % cols > 0)
    fig = plt.figure(figsize=(15, rows * 4))
    fig.suptitle(title, fontsize=16)
    for i, image in enumerate(images):
        ax = fig.add_subplot(rows, cols, i + 1)
        img = image.numpy().transpose(1, 2, 0)  # Convert tensor image to numpy array and change dimensions.
        img = (img - img.min()) / (img.max() - img.min())  # Normalize the image to [0, 1] range.
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Frame {i + 1}", fontsize=12)  # Add frame number as title.

    plt.show()


if __name__ == '__main__':
    # Usage of the dataset.
    dataset = VideoFrameDataset(
        root_dir='data/processed  data',
        transform=Compose([
            Resize((224, 224)),  # Resize images to 224x224.
        ])
    )
    print("Dataset samples: {}".format(len(dataset)))

    # Select a sample from the dataset.
    sample_idx = 0  # You can change this index to view different samples.
    sample = dataset[sample_idx]
    print("Sample shape:", sample['data'].shape)
    images = sample['data']

    # Unbind the frames from the sample tensor.
    frames = list(images.unbind(dim=1))
    # Visualize the frames.
    show_images(frames, cols=4,  title="noraml")  # Adjust cols as needed for better display.


    # 加噪声
    sample_idx = 1  # You can change this index to view different samples.
    sample = dataset[sample_idx]
    images = sample['data']
    # Unbind the frames from the sample tensor.
    frames = list(images.unbind(dim=1))
    # Visualize the frames.
    show_images(frames, cols=4,  title='noise')  # Adjust cols as needed for better display.


    # 镜像增强
    sample_idx = 2  # You can change this index to view different samples.
    sample = dataset[sample_idx]
    images = sample['data']
    # Unbind the frames from the sample tensor.
    frames = list(images.unbind(dim=1))
    # Visualize the frames.
    show_images(frames, cols=4,  title='clip')  # Adjust cols as needed for better display.
