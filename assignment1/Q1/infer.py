import argparse
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data.iloc[index, 1:].values.astype('uint8').reshape((3, 32, 32))
        label = self.data.iloc[index, 0]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the normalization methods
def get_normalization(normalization):
    if normalization == 'bn':
        return torch.nn.BatchNorm2d
    elif normalization == 'in':
        return torch.nn.InstanceNorm2d
    elif normalization == 'bin':
        return torch.nn.BatchNorm2d
    elif normalization == 'ln':
        return torch.nn.LayerNorm
    elif normalization == 'gn':
        return torch.nn.GroupNorm
    elif normalization == 'nn':
        return None
    else:
        raise ValueError('Invalid normalization method: {}'.format(normalization))

# Define the inference function
def infer(model_file, normalization, n, test_data_file, output_file):
    # Load the model
    model = torch.load(model_file)
    model.eval()

    # Set up the device for inference
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set up the normalization layer
    if normalization is not None:
        norm_layer = get_normalization(normalization)(n)
        norm_layer.to(device)
    else:
        norm_layer = None

    # Define the data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Create the dataset and data loader
    dataset = CustomDataset(test_data_file, transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Perform inference on the test data
    with torch.no_grad():
        results = []
        for images, labels in data_loader:
            images = images.to(device)
            if norm_layer is not None:
                images = norm_layer(images)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            results.extend(predicted.cpu().numpy().tolist())

    # Write the results to the output file
    output_df = pd.DataFrame({'Label': results})
    output_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_file', type=str, required=True, help='Path to the model file')
    parser.add_argument('--normalization', type=str, default=None, help='Normalization method')
    parser.add_argument('--n', type=int, default=None, help='Parameter for normalization method')
    parser.add_argument('--test_data_file', type=str, required=True, help='Path to the test data file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
    args = parser.parse_args()

    infer(args.model_file, args.normalization, args.n, args.test_data_file, args.output_file)
