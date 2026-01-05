import os
import argparse
from argparse import Namespace
from pathlib import Path
from src import utils, model, data_loader


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(Path(__file__).parent / "datasets" / "fashion-product-images-small")
    )
    args: Namespace = parser.parse_args()

    # 1. Define the Model Directory
    model_dir: str = os.getenv("AIP_MODEL_DIR", "deploy/saved_models")

    # 2. Load Data via Protocol
    loader: utils.DataLoader = data_loader.get_loader(
        base_path=args.data_path,
    )
    X_train, X_val, X_test, y_train, y_val, y_test, classes = loader.load_and_split(seed=42)

    # 3. Build Model via Protocol
    cnn_model: utils.DeepLearningModel = model.create_model(
        learning_rate=0.001,
        width=60,
        height=80,
        depth=3,
        classes=len(classes),
    )

    # 4. Train
    cnn_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # 5. Evaluate and Save
    loss, acc = cnn_model.evaluate(X_test, y_test, batch_size=args.batch_size)
    print(f"Test Accuracy: {acc:.2f}")
    cnn_model.save(f"{model_dir}/multi_label_classifier.keras")

if __name__ == "__main__":
    main()