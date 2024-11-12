import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Paths to the training and validation data directories
train_dir = '/workspaces/school-cheating-ai-using-cv/new_data_copy'
val_dir = '/workspaces/school-cheating-ai-using-cv/new_data_copy'

# Set the image size (224x224 is the default for MobileNetV2)
img_size = (224, 224)

# Load pre-trained MobileNetV2 model without the top (classification) layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of the pre-trained base model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global average pooling layer
x = Dropout(0.5)(x)  # Dropout layer to prevent overfitting
x = Dense(1024, activation='relu')(x)  # Fully connected layer
x = BatchNormalization()(x)  # Batch normalization for better training
x = Dense(2, activation='softmax')(x)  # Final output layer (assuming 2 classes)

# Define the full model
model = Model(inputs=base_model.input, outputs=x)

# Freeze the base model layers for transfer learning
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up ImageDataGenerators for training and validation data
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize images to [0, 1]
    rotation_range=40,  # Random rotations
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,  # Random shear
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Fill missing pixels after transformation
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for validation data

# Flow images from directories (adjust the paths to your dataset)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,  # Resize images to 224x224
    batch_size=32,
    class_mode='categorical'  # For multi-class classification
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical'
)

# Set up learning rate scheduler
lr_scheduler = ReduceLROnPlateau(factor=0.1, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[lr_scheduler]
)

# After initial training, unfreeze some layers of the base model and continue training
for layer in base_model.layers[:100]:  # Freeze the first 100 layers
    layer.trainable = False
for layer in base_model.layers[100:]:  # Unfreeze the remaining layers
    layer.trainable = True

# Recompile the model with a lower learning rate and fine-tune
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continue training for more epochs with fine-tuning
history_fine_tune = model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(val_generator)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Save the trained model
model.save('image_classification_model.h5')
