import os
import json
import numpy as np
import argparse
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
from typing import Optional, Dict, Any, Tuple, List

def load_model_from_json_and_weights(
    json_path: str, weights_path: str
) -> Optional[keras.Model]:
    if not os.path.exists(json_path):
        print(f"Error: Model JSON file not found at {json_path}")
        return None
    if not os.path.exists(weights_path):
         print(f"Error: Model weights file not found at {weights_path}")
         return None

    try:
        print(f"Loading model architecture from: {json_path}")
        with open(json_path, 'r') as json_file:
            loaded_model_json = json_file.read()

        model = model_from_json(loaded_model_json)
        print(f"Loading model weights from: {weights_path}")
        model.load_weights(weights_path)

        print("Model loaded successfully.")
        return model
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def prepare_data_generator(
    test_dir: str, img_width: int, img_height: int, batch_size: int
) -> Tuple[Optional[ImageDataGenerator], Optional[Dict[int, str]]]:
    if not os.path.isdir(test_dir):
        print(f"Error: Test directory not found or is not a directory: {test_dir}")
        return None, None

    print("Using rescale=1./255 for test data preprocessing.")
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    try:
        print(f"Setting up test generator for directory: {test_dir}")
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        if not test_generator.filenames:
             print(f"Warning: No images found in the test directory: {test_dir}")
        elif not test_generator.class_indices:
             print(f"Warning: No class subdirectories found in: {test_dir}")
             return None, None

        class_indices = test_generator.class_indices
        index_to_class = {v: k for k, v in class_indices.items()}
        print(f"Found classes: {index_to_class}")

        return test_generator, index_to_class

    except Exception as e:
        print(f"Error preparing data generator: {e}")
        return None, None

def predict_classes(
    model: keras.Model,
    test_generator: ImageDataGenerator,
    index_to_class: Dict[int, str]
) -> Dict[str, Any]:
    results = {
        'filenames': [],
        'predicted_classes': [],
        'raw_predictions': []
    }
    if test_generator is None or test_generator.samples == 0:
        print("Skipping prediction as test generator is invalid or empty.")
        return results

    test_generator.reset()
    print(f"Predicting on {test_generator.samples} test images...")

    predictions = model.predict(test_generator, verbose=1)

    predicted_class_indices = np.argmax(predictions, axis=1)

    predicted_labels = [index_to_class.get(idx, f"Unknown Index: {idx}") for idx in predicted_class_indices]

    results = {
        'filenames': test_generator.filenames,
        'predicted_classes': predicted_labels,
        'raw_predictions': predictions.tolist()
    }

    return results

def display_predictions(results: Dict[str, Any]) -> None:
    print("\n--- Prediction Results ---")
    if not results['filenames']:
        print("No predictions to display.")
        return

    for filename, predicted_class in zip(results['filenames'], results['predicted_classes']):
        base_filename = os.path.basename(filename)
        print(f"File: {filename:<40} -> Predicted Class: {predicted_class}")

def main(args: argparse.Namespace) -> None:
    model = load_model_from_json_and_weights(args.model_json, args.model_weights)
    if model is None:
        print("Exiting due to model loading failure.")
        return

    test_generator, index_to_class = prepare_data_generator(
        args.test_data_dir,
        args.img_width,
        args.img_height,
        args.batch_size
    )

    if test_generator is None or index_to_class is None:
        print("Exiting due to data preparation failure.")
        return

    results = predict_classes(model, test_generator, index_to_class)

    display_predictions(results)

    if results['predicted_classes']:
        print(f"\nExample - First predicted class: {results['predicted_classes'][0]} for file {results['filenames'][0]}")
    else:
        print("\nNo predictions were made (e.g., empty test directory).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a Keras model and predict image classes.")
    parser.add_argument('--model-json', type=str, default='model_in_json.json',
                        help='Path to the model architecture JSON file.')
    parser.add_argument('--model-weights', type=str, default='model_weights.h5',
                        help='Path to the model weights H5 file.')
    parser.add_argument('--test-data-dir', type=str, default='Test_dir/',
                        help='Path to the test data directory (with class subfolders).')
    parser.add_argument('--img-width', type=int, default=224,
                        help='Target image width.')
    parser.add_argument('--img-height', type=int, default=224,
                        help='Target image height.')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for prediction.')

    parsed_args = parser.parse_args()
    main(parsed_args)
