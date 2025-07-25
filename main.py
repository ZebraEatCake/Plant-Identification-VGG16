import tensorflow as tf
import numpy as np
import os
import random
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from pathlib import Path

# Load models
model_paths = {
    "planttype": "best_model_planttype_83.keras",
    "plantdisease": "best_model_disease_70.keras"
}

# Class labels
class_labels = {
    "planttype": ['Aglaonema', 'Monstera Deliciosa', 'Snake Plant (Dracaena trifasciata)'],
    "plantdisease": ['Brown Streak', 'Green Mottle', 'Healthy']
}

# Tips for disease
plant_tips = {
    "Brown Streak": [
        "Remove and destroy infected plants to prevent spread.",
        "Use disease-free planting materials from certified sources.",
        "Rotate crops to reduce the buildup of the virus in soil.",
        "Control whiteflies, which are common vectors of the virus."
    ],
    "Green Mottle": [
        "Isolate infected plants to limit virus transmission.",
        "Control whiteflies using insecticidal soap or neem oil.",
        "Use virus-resistant tomato varieties if available.",
        "Avoid planting near virus-prone areas or during high vector activity."
    ],
    "Healthy": [
        "Regularly inspect plants for signs of pests or diseases.",
        "Apply balanced fertilizer to support root and leaf development.",
        "Ensure proper spacing for airflow and sunlight exposure.",
        "Use mulching to retain soil moisture and suppress weeds."
    ]
}

# Tips for plant types
plant_type_tips = {
    "Aglaonema": [
        "Keep in low to moderate indirect light—perfect for offices or shaded rooms.",
        "Water when the top 1-2 inches of soil feels dry.",
        "Avoid cold drafts; Aglaonema prefers warm, humid environments.",
        "Wipe leaves occasionally to prevent dust buildup and pests."
    ],
    "Monstera Deliciosa": [
        "Place in bright, indirect light to encourage leaf splits.",
        "Water when the top third of soil is dry; do not overwater.",
        "Provide a moss pole or trellis to support climbing growth.",
        "Wipe large leaves to keep them healthy and photosynthesizing well."
    ],
    "Snake Plant (Dracaena trifasciata)": [
        "Tolerates low light but thrives in bright, indirect light.",
        "Allow soil to dry out completely between watering—very drought-tolerant.",
        "Avoid cold temperatures below 10°C (50°F).",
        "Use a well-draining soil mix to prevent root rot."
    ]
}

# Prediction function
def classify_image(model, img_path, labels, task, include_tip=False):
    img = image.load_img(img_path, target_size=(244, 244))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.imagenet_utils.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    #prediction results
    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions)
    predicted_class = labels[predicted_index]
    confidence = float(np.max(predictions))

    # Include tips
    if include_tip:
        # Map plant tips based on classificaiton tasks
        tip_dict = plant_tips if task == "plantdisease" else plant_type_tips
        tip = random.choice(tip_dict.get(predicted_class, ["No tip available."]))
    else:
        tip = ""

    return os.path.basename(img_path), predicted_class, confidence, tip

# Main loop
while True:
    task = input("\nDo you want to classify 'planttype' or 'plantdisease'? (type 'exit' to quit): ").strip().lower()
    if task == 'exit':
        break
    if task not in model_paths:
        print("Invalid choice. Try again.")
        continue

    model = load_model(model_paths[task])
    labels = class_labels[task]

    folder_name = f"input_{task}"
    folder = Path(folder_name)
    if not folder.exists():
        print(f"Folder '{folder_name}' not found.")
        continue

    image_paths = list(folder.glob("*.jpg"))
    if not image_paths:
        print(f"No .jpg images found in '{folder_name}'.")
        continue

    # Print result
    print("\nPrediction Results:")
    print("{:<25} {:<30} {:<15} {}".format("Image Name", "Predicted Class", "Confidence", "Tips"))
    print("-" * 120)

    total_start = time.time()
    for img_path in image_paths:
        name, predicted_class, confidence, tip = classify_image(model, str(img_path), labels, task, include_tip=True)
        print("{:<25} {:<30} {:<15} {}".format(name, predicted_class, f"{confidence:.2%}", tip))
    total_end = time.time()

    print(f"\nTotal Processing Time: {total_end - total_start:.2f} seconds")

    again = input("Do you want to classify again? (yes/no): ").strip().lower()
    if again != 'yes':
        break

print("\n Program ended.")
