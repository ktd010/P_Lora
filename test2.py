import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from minlora import add_lora, get_lora_params
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch.optim as optim
from tqdm import tqdm

def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained Vision Transformer model directly to the device
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", 
        attn_implementation="sdpa", 
        torch_dtype=torch.float32
    ).to(device)  # Move to device immediately after loading

    # Adjust classifier for 100 classes
    model.classifier = nn.Linear(model.classifier.in_features, 100).to(device)

    # Add LoRA layers to the model
    add_lora(model)

    # Freeze all parameters except LoRA
    for param in model.parameters():
        param.requires_grad = False

    # Enable only LoRA parameters for training
    for param in get_lora_params(model):
        param.requires_grad = True

    # Dataset setup
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset_path = '/mnt/c/Users/kdtar/Kasun_stuff/My_datasets/mini_imgenet'
    dataset = ImageFolder(root=dataset_path, transform=transform)

    # Split dataset into training and validation subsets
    total_size = len(dataset)
    subset_size = int(0.1 * total_size)  # Use 1% of the total dataset
    train_size = int(0.8 * subset_size)
    val_size = subset_size - train_size

    subset_dataset, _ = random_split(dataset, [subset_size, total_size - subset_size])
    train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=4)

    # Optimizer and loss setup
    optimizer = optim.AdamW(get_lora_params(model), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_train
        epoch_acc = 100 * correct_train / total_train
        print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}%")

        # Validation loop
        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).logits
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = running_val_loss / total_val
        val_acc = 100 * correct_val / total_val
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

    print("Training complete.")

if __name__ == '__main__':
    main()
