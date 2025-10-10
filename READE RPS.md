Rock Paper Scissors Image Classifier ✋✂️🪨

This project uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify images of rock, paper, and scissors. It was inspired by the classic hand game and serves as a fun way to practice building and training CNNs from scratch.

📂 Dataset

The dataset is taken from the Rock Paper Scissors dataset provided by Laurence Moroney .

It contains images of hand gestures representing rock, paper, and scissors.

Each image is 150x150 RGB.

Dataset structure:

rps/ rock/ paper/ scissors/ rps-test-set/ rock/ paper/ scissors/

🧠 Model Architecture

The model is a deep CNN built from scratch with the following layers:

Rescaling layer for normalization

4 × Conv2D + MaxPooling2D layers

Flatten + Dense(512) + Dropout(0.5)

Output layer: Dense(3, activation='softmax')

Data augmentation was applied to increase model robustness:

Random flips, rotations, zooms, translations, and contrast changes.

⚙️ Training

The model was compiled and trained with:

loss = 'categorical_crossentropy' optimizer = 'rmsprop' metrics = ['accuracy'] epochs = 25

Training and validation datasets were created using:

tf.keras.utils.image_dataset_from_directory()

Both datasets were cached, shuffled, and prefetched for optimal performance.

📊 Results

After training for 25 epochs, the model achieved:

High training accuracy

Strong validation accuracy

Smooth convergence in both loss and accuracy curves

You can visualize training progress with:

plot_loss_acc(history)

Output example:

(Insert a screenshot of your training/validation accuracy plot here if you want)

🖼️ Example Predictions

To test the model on a single image:

img = image.load_img('example.jpg', target_size=(150, 150)) img_array = image.img_to_array(img) img_array = np.expand_dims(img_array, axis=0) / 255.0 prediction = model_with_aug.predict(img_array)

Output (example):

Predicted: Scissors ✂️

🚀 How to Run

Clone this repository:

git clone https://github.com//rock-paper-scissors-cnn.git cd rock-paper-scissors-cnn

Install dependencies:

pip install tensorflow matplotlib numpy

Place the dataset in the rps/ and rps-test-set/ directories.

Run the training script:

python rock_paper_scissors.py

📚 Key Concepts Practiced

Image classification using CNNs

Data augmentation for better generalization

Train/validation dataset pipelines with TensorFlow

Visualizing training and validation metrics

**Author ✍️: Osama Al Attar
