import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 5
MODEL_SAVE_PATH = 'mnist_cnn.pth'

def main():
    print(f"Using device: {DEVICE}")

    # --- 1. Load Dataset ---
    # Transformation pipeline: convert images to tensors and normalize them
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Mean and std deviation of MNIST dataset
    ])

    # Download and load the training data
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Download and load the test data
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 2. Initialize Model, Loss, and Optimizer ---
    model = CNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. Training Loop ---
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        model.train() # Set the model to training mode
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move data and targets to the configured device
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward pass
            scores = model(data)
            loss = criterion(scores, targets)

            # Backward pass and optimization
            optimizer.zero_grad() # Clear gradients from previous step
            loss.backward()       # Backpropagate the loss
            optimizer.step()      # Update the weights

            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # --- 4. Evaluate the Model ---
    print("\nEvaluating model on the test set...")
    model.eval() # Set the model to evaluation mode
    num_correct = 0
    num_samples = 0
    with torch.no_grad(): # No need to calculate gradients during evaluation
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            scores = model(x)
            _, predictions = scores.max(1) # Get the index of the max log-probability
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    accuracy = (num_correct / num_samples) * 100
    print(f'Accuracy on the test set: {accuracy:.2f}%')

    # --- 5. Save the Model ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()
