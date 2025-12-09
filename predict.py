import torch
from torchvision import datasets, transforms
from model import CNN
import random

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_LOAD_PATH = 'mnist_cnn.pth'

def predict():
    """
    Loads the trained model and makes a prediction on a random image from the test set.
    """
    print(f"Using device: {DEVICE}")

    # --- 1. Load Model ---
    try:
        model = CNN().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=DEVICE))
        model.eval() # Set the model to evaluation mode
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_LOAD_PATH}'.")
        print("Please run train.py first to train and save the model.")
        return

    # --- 2. Load Test Data ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # --- 3. Make a Prediction on a Random Image ---
    print("\nMaking a prediction on a random image from the test set...")
    with torch.no_grad():
        # Get a random image and its label from the test set
        index = random.randint(0, len(test_dataset) - 1)
        image, label = test_dataset[index]
        
        # Add a batch dimension (B, C, H, W) and send to the configured device
        image_tensor = image.unsqueeze(0).to(DEVICE)

        # Get the model's raw output (logits)
        output = model(image_tensor)
        
        # Get the index of the highest logit, which corresponds to the predicted class
        _, predicted_class = torch.max(output, 1)

        print(f"  - Actual Label:    {label}")
        print(f"  - Predicted Label: {predicted_class.item()}")

        # Optional: For visual confirmation, you can use matplotlib.
        # You would need to install it first (`pip install matplotlib`).
        # import matplotlib.pyplot as plt
        # plt.imshow(image.squeeze(), cmap='gray')
        # plt.title(f'Actual: {label}, Predicted: {predicted_class.item()}')
        # plt.show()


if __name__ == '__main__':
    predict()
