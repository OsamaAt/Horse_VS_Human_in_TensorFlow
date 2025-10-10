🐶😺 Cats vs Dogs Classifier using InceptionV3
This project uses Transfer Learning with InceptionV3 to classify images of cats and dogs.

📂 Dataset
The dataset used is Cats and Dogs Filtered from TensorFlow:

Train and validation folders are structured as: cats_and_dogs_filtered/ train/ cats/ dogs/ validation/ cats/ dogs/
⚙️ Model Architecture
Base Model: InceptionV3 (without top layers)
Custom Layers:
Flatten
Dense(512, ReLU)
Dropout(0.3)
Dense(1, Sigmoid)
🧠 Training
Optimizer: RMSprop (lr=0.0001)
Loss: Binary Crossentropy
Metrics: Accuracy
Epochs: 20
Data Augmentation applied (flip, rotation, zoom, translation, contrast)
📊 Results
After training, the model reached a high validation accuracy with smooth convergence:

Metric	Training	Validation
Accuracy	~0.95	~0.90
🖼️ Testing with Single Image
You can test any image (cat or dog)

🧰 Requirements nginx Copy code tensorflow matplotlib numpy 🏁 How to Run Clone this repository.

Download the pretrained weights file: inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

Place it in your dataset folder.

Run the notebook or Python script.

📸 Sample Predictions Input Output 🐈 Cat 😺 🐕 dog 🐶

