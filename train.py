"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, TinyVGG_model_builder, utils
from torchvision import transforms
import argparse

def main(args):
    # Setup directories
    train_dir = args.train_dir
    test_dir = args.test_dir

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create transforms
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=args.batch_size
    )

    # Create model with help from model_builder.py
    model = TinyVGG_model_builder.TinyVGG(
        input_shape=3,
        hidden_units=args.hidden_units,
        output_shape=len(class_names)
    ).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate)

    # Start training with help from engine.py
    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 epochs=args.num_epochs,
                 device=device)

    # Save the model with help from utils.py
    utils.save_model(model=model,
                     target_dir="models",
                     model_name="tinyvgg_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PyTorch image classification model.")
    
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--hidden_units', type=int, default=10, help='Number of hidden units in the model.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--train_dir', type=str, default="data/pizza_steak_sushi/train", help='Directory for training data.')
    parser.add_argument('--test_dir', type=str, default="data/pizza_steak_sushi/test", help='Directory for test data.')
    
    args = parser.parse_args()
    main(args)
