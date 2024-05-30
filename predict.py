"""
Uses the trained model to predict the class of a given image.
"""

import torch
from torchvision import transforms
from PIL import Image
import sys
import argparse
import TinyVGG_model_builder

def main(args):
    # Define the path to your saved model
    model_path = args.model_dir
    image_path = args.image_dir

    # Setup target device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transformation to apply to the input image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Load the saved model state dict
    model_state_dict = torch.load(model_path)

    # Create an instance of the model and load the state dictionary
    model = TinyVGG_model_builder.TinyVGG(
        input_shape=3,
        hidden_units=10,
        output_shape=3
    )
    model.load_state_dict(model_state_dict)

    model.to(device)

    model.eval()

    # Function to predict on a single image
    def predict(image_path):
        # Load the image
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)

        # Send the image to the target device
        image = image.to(device)

        # Make the prediction
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        return predicted.item()

    # Call the predict function and print the prediction
    prediction = predict(image_path)
    print(f"The predicted class is: {prediction}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the class of an image using a trained PyTorch model.")
    
    parser.add_argument('--model_dir', type=str, default="models/tinyvgg_model.pth", help='Directory for trained model.')
    parser.add_argument('--image_dir', type=str, default="predict/to_predict.jpg", help='Directory for image to predict.')
    
    args = parser.parse_args()
    main(args)