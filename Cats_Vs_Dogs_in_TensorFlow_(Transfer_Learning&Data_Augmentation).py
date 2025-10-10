import tensorflow as tf 
import matplotlib.pyplot as plt 
import os 
from keras.preprocessing import image 
import numpy as np 

local_weights_file='C:\X\Deep_Learning\Datasets for CNN\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model=tf.keras.applications.inception_v3.InceptionV3(
    input_shape=(150,150,3),
    weights=None,
    include_top=False)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable=False

pre_trained_model.summary()

#choose 'mixed7' as last layer of your base model 
last_layer=pre_trained_model.get_layer('mixed7')
print(f'last layer output shape : {last_layer.output.shape}')
last_output=last_layer.output

#Now we will add Dense Layers to our model These will be the layers that you will train and is tasked with recognizing cats and dogs

x=tf.keras.layers.Flatten()(last_output)

x=tf.keras.layers.Dense(512 , activation='relu')(x)

x=tf.keras.layers.Dropout(0.3)(x)

x=tf.keras.layers.Dense(1,activation='sigmoid')(x)

#Append the dense netowrk to base model
model=tf.keras.Model(pre_trained_model.input , x)

model.summary()

BASE_DIR='C:\X\Deep_Learning\Datasets for CNN\cats_and_dogs_filtered'

train_dir=os.path.join(BASE_DIR , 'train')
validation_dir=os.path.join(BASE_DIR , 'validation')

train_cats_dir=os.path.join(train_dir,'cats')
train_dogs_dir=os.path.join(train_dir,'dogs')

validation_cats_dir=os.path.join(validation_dir , 'cats')
validation_dogs_dir=os.path.join(validation_dir , 'dogs')

train_dataset=tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(150,150),
    batch_size=20,
    label_mode='binary'
)

validation_dataset=tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=(150,150),
    batch_size=20,
    label_mode='binary'
)
def preprocess(image, label):
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image, label

# Apply the preprocessing to the datasets
train_dataset_scaled = train_dataset.map(preprocess)
validation_dataset_scaled = validation_dataset.map(preprocess)

SUFFLE_BUFEER_SIZE=1000
PREFETCH_BUFFER_SIZE=tf.data.AUTOTUNE

train_dataset_final=(train_dataset_scaled.cache().shuffle(SUFFLE_BUFEER_SIZE).prefetch(PREFETCH_BUFFER_SIZE))
validation_dataset_final=(validation_dataset_scaled.cache().prefetch(PREFETCH_BUFFER_SIZE))

data_augmentation=tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.4,),
    tf.keras.layers.RandomTranslation(0.2,0.2),
    tf.keras.layers.RandomContrast(0.4),
    tf.keras.layers.RandomZoom(0.2),
])

inputs=tf.keras.Input(shape=(150,150,3))
x=data_augmentation(inputs)
x=model(x)

model_with_aug=tf.keras.Model(inputs , x )

model_with_aug.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history=model_with_aug.fit(train_dataset_final , epochs=20 , validation_data=validation_dataset_final , verbose=2)

def plot_loss_acc(history):
    '''Plots the training and validation loss and accuracy from a history object'''
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    
    fig, ax = plt.subplots(1,2, figsize=(12, 6))
    ax[0].plot(epochs, acc, 'bo', label='Training accuracy')
    ax[0].plot(epochs, val_acc, 'b', label='Validation accuracy')
    ax[0].set_title('Training and validation accuracy')
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('accuracy')
    ax[0].legend()
    
    ax[1].plot(epochs, loss, 'bo', label='Training Loss')
    ax[1].plot(epochs, val_loss, 'b', label='Validation Loss')
    ax[1].set_title('Training and validation loss')
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('loss')
    ax[1].legend()
    
    plt.show()

plot_loss_acc(history)

# ---- Test single image ----
img_path = r"C:\Users\Asus\Downloads\download.jpg"  # Change this
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Rescale

prediction = model_with_aug.predict(img_array)

if prediction[0][0] > 0.5:
    print("Predicted: Dog 🐶")
else:
    print("Predicted: Cat 😺")
