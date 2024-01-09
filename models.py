import torchvision
from torch import nn


# Define the r3d_18 class, which is a 3D ResNet model.
class r3d_18(nn.Module):
    def __init__(self, pretrained=True, num_classes=2):
        super(r3d_18, self).__init__()
        self.pretrained = pretrained  # Indicate whether to use a pretrained model.
        self.num_classes = num_classes  # Number of output classes.

        # Load the pretrained 3D ResNet model.
        model = torchvision.models.video.r3d_18(pretrained=self.pretrained)

        # Remove the last fully connected (fc) layer from the model.
        modules = list(model.children())[:-1]
        self.r3d_18 = nn.Sequential(*modules)  # Reconstruct the model without the last fc layer.
        self.dropout = nn.Dropout(p=0.2)

        # Add a new fc layer with output size equal to the number of classes.
        self.fc1 = nn.Linear(model.fc.in_features, self.num_classes)

    def forward(self, x):
        # Define the forward pass.
        out = self.r3d_18(x)  # Pass the input through the modified 3D ResNet model.
        out = out.flatten(1)  # Flatten the output for the fc layer.
        out = self.fc1(out)  # Pass the output through the new fc layer.
        return out


# Define the r2plus1d_18 class, which is a R(2+1)D model.
class r2plus1d_18(nn.Module):
    def __init__(self, pretrained=True, num_classes=2):
        super(r2plus1d_18, self).__init__()
        self.pretrained = pretrained  # Indicate whether to use a pretrained model.
        self.num_classes = num_classes  # Number of output classes.

        # Load the pretrained R(2+1)D model.
        model = torchvision.models.video.r2plus1d_18(pretrained=pretrained)

        # Remove the last fully connected (fc) layer from the model.
        modules = list(model.children())[:-1]
        self.r2plus1d_18 = nn.Sequential(*modules)  # Reconstruct the model without the last fc layer.

        # Add a new fc layer with output size equal to the number of classes.
        self.fc1 = nn.Linear(model.fc.in_features, self.num_classes)

    def forward(self, x):
        # Define the forward pass.
        out = self.r2plus1d_18(x)  # Pass the input through the modified R(2+1)D model.
        out = out.flatten(1)  # Flatten the output for the fc layer.
        out = self.fc1(out)  # Pass the output through the new fc layer.
        return out


models = {
    'r3d_18': r3d_18,
    'r2plus1d_18': r2plus1d_18
}


def get_model(model_name, pretrained=True, num_classes=2):
    model = models[model_name](pretrained=pretrained, num_classes=num_classes)
    return model
