# ML Implementation Guide for Skin Cancer Detection

This guide provides comprehensive documentation for implementing the machine learning pipeline for the SkinGuard AI skin cancer detection system.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Data Pipeline](#data-pipeline)
3. [Model Implementation](#model-implementation)
4. [Training Pipeline](#training-pipeline)
5. [Inference Integration](#inference-integration)
6. [Performance Optimization](#performance-optimization)
7. [Deployment Strategy](#deployment-strategy)

## Architecture Overview

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Mobile App    │    │   Data API      │    │  ML Pipeline    │
│                 │    │                 │    │                 │
│ • Image Capture │───▶│ • Data Storage  │───▶│ • Preprocessing │
│ • User Interface│    │ • Annotation    │    │ • Training      │
│ • Results       │◀───│ • Validation    │◀───│ • Inference     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Technology Stack

- **Frontend**: React Native (Expo)
- **Backend**: Node.js/Python FastAPI
- **ML Framework**: TensorFlow/PyTorch
- **Data Storage**: Cloud Storage + Database
- **Model Serving**: TensorFlow Serving/ONNX Runtime

## Data Pipeline

### 1. Data Collection Integration

The app's data collection system is already implemented. Here's how to extend it:

#### Database Schema

```sql
-- Images table
CREATE TABLE training_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    image_url TEXT NOT NULL,
    image_hash TEXT UNIQUE NOT NULL,
    metadata JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Annotations table
CREATE TABLE image_annotations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_id UUID REFERENCES training_images(id),
    label VARCHAR(20) NOT NULL CHECK (label IN ('benign', 'malignant')),
    confidence FLOAT,
    annotator_id UUID,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Training datasets
CREATE TABLE training_datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    image_count INTEGER DEFAULT 0,
    train_split FLOAT DEFAULT 0.8,
    val_split FLOAT DEFAULT 0.1,
    test_split FLOAT DEFAULT 0.1,
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### API Endpoints to Implement

```typescript
// POST /api/images/upload
interface ImageUploadRequest {
  image: File;
  metadata: {
    location?: string;
    size?: string;
    color?: string;
    notes?: string;
  };
}

// POST /api/images/:id/annotate
interface AnnotationRequest {
  label: 'benign' | 'malignant';
  confidence?: number;
  metadata?: Record<string, any>;
}

// GET /api/datasets/:id/export
interface DatasetExportResponse {
  images: Array<{
    id: string;
    url: string;
    label: string;
    metadata: Record<string, any>;
  }>;
  splits: {
    train: string[];
    validation: string[];
    test: string[];
  };
}
```

### 2. Data Preprocessing Pipeline

#### Image Preprocessing

```python
import cv2
import numpy as np
from PIL import Image
import albumentations as A

class ImagePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.augmentation_pipeline = A.Compose([
            A.Resize(target_size[0], target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path: str, augment: bool = False) -> np.ndarray:
        """
        Preprocess a single image for training or inference
        
        Args:
            image_path: Path to the image file
            augment: Whether to apply data augmentation
            
        Returns:
            Preprocessed image as numpy array
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing
        if augment:
            augmented = self.augmentation_pipeline(image=image)
            image = augmented['image']
        else:
            # Just resize and normalize for inference
            image = cv2.resize(image, self.target_size)
            image = image.astype(np.float32) / 255.0
            # Normalize with ImageNet stats
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
        
        return image
    
    def create_dataset(self, image_paths: list, labels: list, batch_size: int = 32):
        """
        Create a TensorFlow dataset from image paths and labels
        """
        import tensorflow as tf
        
        def load_and_preprocess(path, label):
            image = tf.py_function(
                func=self.preprocess_image,
                inp=[path, True],  # Apply augmentation
                Tout=tf.float32
            )
            image.set_shape([self.target_size[0], self.target_size[1], 3])
            return image, label
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
```

#### Data Quality Validation

```python
class DataQualityValidator:
    def __init__(self):
        self.min_resolution = (224, 224)
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.allowed_formats = ['.jpg', '.jpeg', '.png']
    
    def validate_image(self, image_path: str) -> dict:
        """
        Validate image quality and return metrics
        
        Returns:
            Dictionary with validation results and quality metrics
        """
        try:
            # Check file format
            if not any(image_path.lower().endswith(fmt) for fmt in self.allowed_formats):
                return {'valid': False, 'error': 'Invalid file format'}
            
            # Check file size
            file_size = os.path.getsize(image_path)
            if file_size > self.max_file_size:
                return {'valid': False, 'error': 'File too large'}
            
            # Load and check image
            image = cv2.imread(image_path)
            if image is None:
                return {'valid': False, 'error': 'Cannot read image'}
            
            height, width = image.shape[:2]
            
            # Check resolution
            if width < self.min_resolution[0] or height < self.min_resolution[1]:
                return {'valid': False, 'error': 'Resolution too low'}
            
            # Calculate quality metrics
            blur_score = self._calculate_blur_score(image)
            brightness_score = self._calculate_brightness_score(image)
            contrast_score = self._calculate_contrast_score(image)
            
            return {
                'valid': True,
                'metrics': {
                    'resolution': (width, height),
                    'file_size': file_size,
                    'blur_score': blur_score,
                    'brightness_score': brightness_score,
                    'contrast_score': contrast_score
                }
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _calculate_blur_score(self, image):
        """Calculate Laplacian variance to detect blur"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def _calculate_brightness_score(self, image):
        """Calculate average brightness"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def _calculate_contrast_score(self, image):
        """Calculate RMS contrast"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.std(gray)
```

## Model Implementation

### 1. Model Architecture

#### Base CNN Model

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class SkinCancerCNN:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build_model(self) -> Model:
        """
        Build a CNN model for skin cancer classification
        
        Architecture:
        - Convolutional layers with batch normalization
        - MaxPooling for downsampling
        - Dropout for regularization
        - Dense layers for classification
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # First convolutional block
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Second convolutional block
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Third convolutional block
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Fourth convolutional block
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Global average pooling and classification
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='skin_cancer_cnn')
        return model
```

#### Transfer Learning Models

```python
class TransferLearningModel:
    def __init__(self, base_model_name='EfficientNetB0', input_shape=(224, 224, 3), num_classes=2):
        self.base_model_name = base_model_name
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build_model(self, trainable_layers=None) -> Model:
        """
        Build a transfer learning model using pre-trained weights
        
        Args:
            trainable_layers: Number of top layers to make trainable (None = all frozen)
        """
        # Load pre-trained base model
        if self.base_model_name == 'EfficientNetB0':
            base_model = tf.keras.applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.base_model_name == 'ResNet50':
            base_model = tf.keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.base_model_name == 'DenseNet121':
            base_model = tf.keras.applications.DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported base model: {self.base_model_name}")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # If specified, make top layers trainable
        if trainable_layers is not None:
            base_model.trainable = True
            for layer in base_model.layers[:-trainable_layers]:
                layer.trainable = False
        
        # Add custom classification head
        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name=f'transfer_{self.base_model_name.lower()}')
        return model
```

### 2. Training Configuration

```python
class TrainingConfig:
    def __init__(self):
        # Model parameters
        self.input_shape = (224, 224, 3)
        self.num_classes = 2
        self.model_type = 'EfficientNetB0'  # 'CNN', 'EfficientNetB0', 'ResNet50', 'DenseNet121'
        
        # Training parameters
        self.batch_size = 32
        self.epochs = 100
        self.learning_rate = 0.001
        self.optimizer = 'adam'  # 'adam', 'sgd', 'rmsprop'
        
        # Data splits
        self.train_split = 0.7
        self.val_split = 0.15
        self.test_split = 0.15
        
        # Callbacks
        self.early_stopping_patience = 10
        self.reduce_lr_patience = 5
        self.reduce_lr_factor = 0.5
        
        # Regularization
        self.dropout_rate = 0.5
        self.l2_regularization = 0.001
        
        # Data augmentation
        self.use_augmentation = True
        self.augmentation_strength = 0.5
        
        # Class balancing
        self.use_class_weights = True
        self.use_focal_loss = False  # Alternative to class weights
        
        # Model checkpointing
        self.save_best_only = True
        self.monitor_metric = 'val_accuracy'
        self.checkpoint_dir = './checkpoints'
        
        # Logging
        self.log_dir = './logs'
        self.log_frequency = 10  # Log every N batches
```

## Training Pipeline

### 1. Training Manager

```python
import os
import json
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class TrainingManager:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.history = None
        self.training_start_time = None
        
        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
    
    def prepare_data(self, dataset_path: str):
        """
        Load and prepare the dataset for training
        
        Args:
            dataset_path: Path to the dataset directory or metadata file
        """
        # Load dataset metadata
        with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Extract image paths and labels
        image_paths = []
        labels = []
        
        for item in metadata['images']:
            image_paths.append(os.path.join(dataset_path, item['path']))
            labels.append(1 if item['label'] == 'malignant' else 0)
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            image_paths, labels, 
            test_size=self.config.test_split, 
            stratify=labels, 
            random_state=42
        )
        
        val_size = self.config.val_split / (1 - self.config.test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size, 
            stratify=y_temp, 
            random_state=42
        )
        
        # Create datasets
        preprocessor = ImagePreprocessor()
        
        self.train_dataset = preprocessor.create_dataset(
            X_train, y_train, self.config.batch_size
        )
        self.val_dataset = preprocessor.create_dataset(
            X_val, y_val, self.config.batch_size
        )
        self.test_dataset = preprocessor.create_dataset(
            X_test, y_test, self.config.batch_size
        )
        
        # Calculate class weights if needed
        if self.config.use_class_weights:
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight(
                'balanced', 
                classes=np.unique(y_train), 
                y=y_train
            )
            self.class_weights = {i: weight for i, weight in enumerate(class_weights)}
        else:
            self.class_weights = None
        
        print(f"Dataset prepared:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Test samples: {len(X_test)}")
        if self.class_weights:
            print(f"  Class weights: {self.class_weights}")
    
    def build_model(self):
        """Build the model based on configuration"""
        if self.config.model_type == 'CNN':
            model_builder = SkinCancerCNN(
                input_shape=self.config.input_shape,
                num_classes=self.config.num_classes
            )
        else:
            model_builder = TransferLearningModel(
                base_model_name=self.config.model_type,
                input_shape=self.config.input_shape,
                num_classes=self.config.num_classes
            )
        
        self.model = model_builder.build_model()
        
        # Compile model
        optimizer = self._get_optimizer()
        loss = self._get_loss_function()
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"Model built: {self.config.model_type}")
        print(f"Total parameters: {self.model.count_params():,}")
    
    def _get_optimizer(self):
        """Get optimizer based on configuration"""
        if self.config.optimizer == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'sgd':
            return tf.keras.optimizers.SGD(
                learning_rate=self.config.learning_rate, 
                momentum=0.9
            )
        elif self.config.optimizer == 'rmsprop':
            return tf.keras.optimizers.RMSprop(learning_rate=self.config.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def _get_loss_function(self):
        """Get loss function based on configuration"""
        if self.config.use_focal_loss:
            # Implement focal loss for handling class imbalance
            def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
                epsilon = tf.keras.backend.epsilon()
                y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
                
                # Calculate focal loss
                alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
                p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
                focal_loss = -alpha_t * tf.pow((1 - p_t), gamma) * tf.log(p_t)
                
                return tf.reduce_mean(focal_loss)
            
            return focal_loss
        else:
            return 'sparse_categorical_crossentropy'
    
    def _get_callbacks(self):
        """Setup training callbacks"""
        callbacks = []
        
        # Model checkpointing
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, 
            f'best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'
        )
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=self.config.monitor_metric,
            save_best_only=self.config.save_best_only,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=self.config.monitor_metric,
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=self.config.monitor_metric,
            factor=self.config.reduce_lr_factor,
            patience=self.config.reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard logging
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(self.config.log_dir, datetime.now().strftime("%Y%m%d_%H%M%S")),
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard_callback)
        
        # Custom logging callback
        class TrainingLogger(tf.keras.callbacks.Callback):
            def __init__(self, log_frequency=10):
                self.log_frequency = log_frequency
                self.batch_count = 0
            
            def on_batch_end(self, batch, logs=None):
                self.batch_count += 1
                if self.batch_count % self.log_frequency == 0:
                    print(f"Batch {self.batch_count}: {logs}")
        
        callbacks.append(TrainingLogger(self.config.log_frequency))
        
        return callbacks
    
    def train(self):
        """Start the training process"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if not hasattr(self, 'train_dataset'):
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        print("Starting training...")
        self.training_start_time = datetime.now()
        
        # Get callbacks
        callbacks = self._get_callbacks()
        
        # Start training
        self.history = self.model.fit(
            self.train_dataset,
            epochs=self.config.epochs,
            validation_data=self.val_dataset,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        
        training_time = datetime.now() - self.training_start_time
        print(f"Training completed in {training_time}")
        
        # Save training history
        self._save_training_history()
        
        # Evaluate on test set
        self.evaluate_model()
    
    def _save_training_history(self):
        """Save training history to file"""
        history_path = os.path.join(
            self.config.log_dir, 
            f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"Training history saved to {history_path}")
    
    def evaluate_model(self):
        """Evaluate the model on test set and generate reports"""
        print("Evaluating model on test set...")
        
        # Get predictions
        y_pred_proba = self.model.predict(self.test_dataset)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Get true labels
        y_true = []
        for _, labels in self.test_dataset:
            y_true.extend(labels.numpy())
        y_true = np.array(y_true)
        
        # Calculate metrics
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            self.test_dataset, verbose=0
        )
        
        print(f"Test Results:")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  Loss: {test_loss:.4f}")
        
        # Classification report
        class_names = ['Benign', 'Malignant']
        report = classification_report(y_true, y_pred, target_names=class_names)
        print("\nClassification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self._plot_confusion_matrix(cm, class_names)
        
        # Training curves
        self._plot_training_curves()
        
        # Save evaluation results
        self._save_evaluation_results(test_accuracy, test_precision, test_recall, test_loss, report)
    
    def _plot_confusion_matrix(self, cm, class_names):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plot_path = os.path.join(
            self.config.log_dir, 
            f'confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {plot_path}")
    
    def _plot_training_curves(self):
        """Plot and save training curves"""
        if self.history is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        plot_path = os.path.join(
            self.config.log_dir, 
            f'training_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {plot_path}")
    
    def _save_evaluation_results(self, accuracy, precision, recall, loss, report):
        """Save evaluation results to file"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'loss': float(loss)
            },
            'classification_report': report,
            'training_time': str(datetime.now() - self.training_start_time)
        }
        
        results_path = os.path.join(
            self.config.log_dir, 
            f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation results saved to {results_path}")
```

### 2. Training Script Example

```python
# train_model.py
"""
Main training script for skin cancer detection model

Usage:
    python train_model.py --dataset_path ./data --model_type EfficientNetB0 --epochs 50
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Train skin cancer detection model')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to the dataset directory')
    parser.add_argument('--model_type', type=str, default='EfficientNetB0',
                       choices=['CNN', 'EfficientNetB0', 'ResNet50', 'DenseNet121'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Use class weights to handle imbalanced data')
    parser.add_argument('--use_focal_loss', action='store_true',
                       help='Use focal loss instead of standard cross-entropy')
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path {args.dataset_path} does not exist")
        sys.exit(1)
    
    # Create training configuration
    config = TrainingConfig()
    config.model_type = args.model_type
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.use_class_weights = args.use_class_weights
    config.use_focal_loss = args.use_focal_loss
    
    print(f"Training configuration:")
    print(f"  Model: {config.model_type}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Use class weights: {config.use_class_weights}")
    print(f"  Use focal loss: {config.use_focal_loss}")
    
    # Initialize training manager
    trainer = TrainingManager(config)
    
    try:
        # Prepare data
        print("Preparing dataset...")
        trainer.prepare_data(args.dataset_path)
        
        # Build model
        print("Building model...")
        trainer.build_model()
        
        # Start training
        print("Starting training...")
        trainer.train()
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Inference Integration

### 1. Model Serving

```python
# model_server.py
"""
FastAPI server for serving the trained model

This server provides REST API endpoints for model inference
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import base64
from typing import Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Skin Cancer Detection API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
preprocessor = None

class ModelInference:
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
        self.preprocessor = ImagePreprocessor()
        self.class_names = ['Benign', 'Malignant']
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Model input shape: {self.model.input_shape}")
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Make prediction on a single image
        
        Args:
            image: Preprocessed image array
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Add batch dimension
            image_batch = np.expand_dims(image, axis=0)
            
            # Make prediction
            predictions = self.model.predict(image_batch, verbose=0)
            
            # Get probabilities and predicted class
            probabilities = predictions[0]
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])
            
            # Prepare result
            result = {
                'predicted_class': self.class_names[predicted_class],
                'confidence': confidence,
                'probabilities': {
                    'benign': float(probabilities[0]),
                    'malignant': float(probabilities[1])
                },
                'risk_level': self._get_risk_level(confidence, predicted_class)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def _get_risk_level(self, confidence: float, predicted_class: int) -> str:
        """
        Determine risk level based on prediction confidence
        
        Args:
            confidence: Prediction confidence
            predicted_class: Predicted class (0=benign, 1=malignant)
            
        Returns:
            Risk level string
        """
        if predicted_class == 0:  # Benign
            if confidence > 0.9:
                return 'low'
            elif confidence > 0.7:
                return 'low-medium'
            else:
                return 'medium'
        else:  # Malignant
            if confidence > 0.8:
                return 'high'
            elif confidence > 0.6:
                return 'medium-high'
            else:
                return 'medium'

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, preprocessor
    
    model_path = os.getenv('MODEL_PATH', './models/best_model.h5')
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise RuntimeError(f"Model file not found: {model_path}")
    
    try:
        model = ModelInference(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict skin cancer from uploaded image
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Validate image quality
        validator = DataQualityValidator()
        
        # Save temporary file for validation
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as temp_file:
            temp_file.write(image_data)
        
        validation_result = validator.validate_image(temp_path)
        os.remove(temp_path)  # Clean up
        
        if not validation_result['valid']:
            raise HTTPException(status_code=400, detail=validation_result['error'])
        
        # Preprocess image
        preprocessed_image = model.preprocessor.preprocess_image_array(image_array)
        
        # Make prediction
        result = model.predict(preprocessed_image)
        
        # Add image quality metrics
        result['image_quality'] = validation_result.get('metrics', {})
        
        logger.info(f"Prediction made: {result['predicted_class']} (confidence: {result['confidence']:.3f})")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_base64")
async def predict_base64_image(data: dict):
    """
    Predict skin cancer from base64 encoded image
    
    Args:
        data: Dictionary containing base64 encoded image
        
    Returns:
        Prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Preprocess image
        preprocessed_image = model.preprocessor.preprocess_image_array(image_array)
        
        # Make prediction
        result = model.predict(preprocessed_image)
        
        logger.info(f"Prediction made: {result['predicted_class']} (confidence: {result['confidence']:.3f})")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. Mobile App Integration

```typescript
// services/MLService.ts
/**
 * Service for integrating with the ML model API
 */

interface PredictionResult {
  predicted_class: 'Benign' | 'Malignant';
  confidence: number;
  probabilities: {
    benign: number;
    malignant: number;
  };
  risk_level: 'low' | 'low-medium' | 'medium' | 'medium-high' | 'high';
  image_quality?: {
    resolution: [number, number];
    file_size: number;
    blur_score: number;
    brightness_score: number;
    contrast_score: number;
  };
}

interface MLConfig {
  apiUrl: string;
  timeout: number;
  retryAttempts: number;
}

class MLService {
  private config: MLConfig;
  
  constructor(config: MLConfig) {
    this.config = config;
  }
  
  /**
   * Predict skin cancer from image URI
   */
  async predictFromImageUri(imageUri: string): Promise<PredictionResult> {
    try {
      // Create form data
      const formData = new FormData();
      
      // Handle different platforms
      if (Platform.OS === 'web') {
        // For web, convert URI to blob
        const response = await fetch(imageUri);
        const blob = await response.blob();
        formData.append('file', blob, 'image.jpg');
      } else {
        // For mobile, use the URI directly
        formData.append('file', {
          uri: imageUri,
          type: 'image/jpeg',
          name: 'image.jpg',
        } as any);
      }
      
      // Make API request
      const response = await this.makeRequest('/predict', {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
      }
      
      const result = await response.json();
      return result;
      
    } catch (error) {
      console.error('Prediction error:', error);
      throw new Error(`Prediction failed: ${error.message}`);
    }
  }
  
  /**
   * Predict skin cancer from base64 image
   */
  async predictFromBase64(base64Image: string): Promise<PredictionResult> {
    try {
      const response = await this.makeRequest('/predict_base64', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: base64Image
        }),
      });
      
      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
      }
      
      const result = await response.json();
      return result;
      
    } catch (error) {
      console.error('Prediction error:', error);
      throw new Error(`Prediction failed: ${error.message}`);
    }
  }
  
  /**
   * Check if the ML service is healthy
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.makeRequest('/health');
      const data = await response.json();
      return data.status === 'healthy' && data.model_loaded;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }
  
  /**
   * Make HTTP request with retry logic
   */
  private async makeRequest(endpoint: string, options: RequestInit = {}): Promise<Response> {
    const url = `${this.config.apiUrl}${endpoint}`;
    
    for (let attempt = 1; attempt <= this.config.retryAttempts; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);
        
        const response = await fetch(url, {
          ...options,
          signal: controller.signal,
        });
        
        clearTimeout(timeoutId);
        return response;
        
      } catch (error) {
        if (attempt === this.config.retryAttempts) {
          throw error;
        }
        
        // Wait before retry
        await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
      }
    }
    
    throw new Error('Max retry attempts reached');
  }
}

// Export singleton instance
export const mlService = new MLService({
  apiUrl: process.env.EXPO_PUBLIC_ML_API_URL || 'http://localhost:8000',
  timeout: 30000, // 30 seconds
  retryAttempts: 3,
});

export type { PredictionResult };
```

### 3. Integration in Camera/Upload Components

```typescript
// Update camera.tsx to use ML service
import { mlService, PredictionResult } from '@/services/MLService';

// In the capturePhoto function:
const capturePhoto = async () => {
  if (isCapturing || !cameraRef.current) return;
  
  try {
    setIsCapturing(true);
    triggerHaptic();
    
    const photo = await cameraRef.current.takePictureAsync({
      quality: 0.8,
      base64: false,
    });
    
    if (photo) {
      // Show loading state
      Alert.alert('Analyzing...', 'Please wait while we analyze your image.');
      
      try {
        // Make prediction
        const result: PredictionResult = await mlService.predictFromImageUri(photo.uri);
        
        // Show results
        const riskMessage = getRiskMessage(result.risk_level);
        const confidencePercent = Math.round(result.confidence * 100);
        
        Alert.alert(
          'Analysis Complete',
          `Result: ${result.predicted_class}\nConfidence: ${confidencePercent}%\n\n${riskMessage}`,
          [
            {
              text: 'Save to History',
              onPress: () => {
                // Save to history with results
                saveToHistory(photo.uri, result);
                router.push('/history');
              },
            },
            {
              text: 'Retake',
              style: 'cancel',
            },
          ]
        );
        
      } catch (error) {
        Alert.alert(
          'Analysis Failed',
          'Unable to analyze the image. Please try again or check your internet connection.',
          [
            {
              text: 'Retry',
              onPress: () => capturePhoto(),
            },
            {
              text: 'Cancel',
              style: 'cancel',
            },
          ]
        );
      }
    }
  } catch (error) {
    Alert.alert('Error', 'Failed to capture photo. Please try again.');
  } finally {
    setIsCapturing(false);
  }
};

function getRiskMessage(riskLevel: string): string {
  switch (riskLevel) {
    case 'low':
      return 'Low risk detected. Continue regular monitoring.';
    case 'low-medium':
      return 'Low to medium risk. Consider consulting a dermatologist.';
    case 'medium':
      return 'Medium risk detected. We recommend seeing a dermatologist.';
    case 'medium-high':
      return 'Medium to high risk. Please consult a dermatologist soon.';
    case 'high':
      return 'High risk detected. Please see a dermatologist immediately.';
    default:
      return 'Analysis complete. Consult a healthcare professional for proper diagnosis.';
  }
}

async function saveToHistory(imageUri: string, result: PredictionResult) {
  // Implementation to save results to local storage or database
  // This would integrate with your data management system
}
```

## Performance Optimization

### 1. Model Optimization

```python
# model_optimization.py
"""
Tools for optimizing trained models for deployment
"""

import tensorflow as tf
import numpy as np
from tensorflow.lite.python import lite

class ModelOptimizer:
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
    
    def convert_to_tflite(self, output_path: str, quantize: bool = True):
        """
        Convert model to TensorFlow Lite format
        
        Args:
            output_path: Path to save the TFLite model
            quantize: Whether to apply quantization
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            # Apply dynamic range quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # For better accuracy, you can use representative dataset
            # converter.representative_dataset = self._representative_dataset_gen
            # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            # converter.inference_input_type = tf.int8
            # converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved to {output_path}")
        
        # Compare model sizes
        original_size = os.path.getsize(self.model_path) / (1024 * 1024)  # MB
        tflite_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        print(f"Original model size: {original_size:.2f} MB")
        print(f"TFLite model size: {tflite_size:.2f} MB")
        print(f"Size reduction: {((original_size - tflite_size) / original_size * 100):.1f}%")
    
    def _representative_dataset_gen(self):
        """
        Generate representative dataset for quantization
        This should use actual training data samples
        """
        # Load a small subset of your training data
        # This is a placeholder - implement with your actual data
        for _ in range(100):
            # Generate random data matching your input shape
            data = np.random.random((1, 224, 224, 3)).astype(np.float32)
            yield [data]
    
    def benchmark_model(self, test_images: np.ndarray, num_runs: int = 100):
        """
        Benchmark model inference speed
        
        Args:
            test_images: Array of test images
            num_runs: Number of inference runs for benchmarking
        """
        import time
        
        # Warm up
        for _ in range(10):
            _ = self.model.predict(test_images[:1], verbose=0)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            _ = self.model.predict(test_images[:1], verbose=0)
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / num_runs
        
        print(f"Average inference time: {avg_inference_time * 1000:.2f} ms")
        print(f"Throughput: {1 / avg_inference_time:.2f} images/second")
        
        return avg_inference_time
```

### 2. Caching and Performance

```typescript
// services/CacheService.ts
/**
 * Service for caching ML predictions and managing performance
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { Platform } from 'react-native';

interface CachedPrediction {
  imageHash: string;
  result: PredictionResult;
  timestamp: number;
  expiryTime: number;
}

class CacheService {
  private static readonly CACHE_KEY = 'ml_predictions_cache';
  private static readonly CACHE_DURATION = 24 * 60 * 60 * 1000; // 24 hours
  private static readonly MAX_CACHE_SIZE = 100; // Maximum cached predictions
  
  /**
   * Generate hash for image to use as cache key
   */
  private async generateImageHash(imageUri: string): Promise<string> {
    if (Platform.OS === 'web') {
      // For web, use a simple hash based on URI and file size
      return btoa(imageUri).replace(/[^a-zA-Z0-9]/g, '').substring(0, 32);
    } else {
      // For mobile, you might want to use a proper hashing library
      // For now, using a simple approach
      return imageUri.split('/').pop()?.replace(/[^a-zA-Z0-9]/g, '') || 'unknown';
    }
  }
  
  /**
   * Get cached prediction if available and not expired
   */
  async getCachedPrediction(imageUri: string): Promise<PredictionResult | null> {
    try {
      const imageHash = await this.generateImageHash(imageUri);
      const cacheData = await AsyncStorage.getItem(this.CACHE_KEY);
      
      if (!cacheData) return null;
      
      const cache: CachedPrediction[] = JSON.parse(cacheData);
      const cachedItem = cache.find(item => item.imageHash === imageHash);
      
      if (!cachedItem) return null;
      
      // Check if cache is expired
      if (Date.now() > cachedItem.expiryTime) {
        // Remove expired item
        await this.removeCachedPrediction(imageHash);
        return null;
      }
      
      return cachedItem.result;
      
    } catch (error) {
      console.error('Error getting cached prediction:', error);
      return null;
    }
  }
  
  /**
   * Cache a prediction result
   */
  async cachePrediction(imageUri: string, result: PredictionResult): Promise<void> {
    try {
      const imageHash = await this.generateImageHash(imageUri);
      const cacheData = await AsyncStorage.getItem(this.CACHE_KEY);
      
      let cache: CachedPrediction[] = cacheData ? JSON.parse(cacheData) : [];
      
      // Remove existing cache for this image
      cache = cache.filter(item => item.imageHash !== imageHash);
      
      // Add new cache entry
      const newCacheItem: CachedPrediction = {
        imageHash,
        result,
        timestamp: Date.now(),
        expiryTime: Date.now() + this.CACHE_DURATION,
      };
      
      cache.unshift(newCacheItem);
      
      // Limit cache size
      if (cache.length > this.MAX_CACHE_SIZE) {
        cache = cache.slice(0, this.MAX_CACHE_SIZE);
      }
      
      await AsyncStorage.setItem(this.CACHE_KEY, JSON.stringify(cache));
      
    } catch (error) {
      console.error('Error caching prediction:', error);
    }
  }
  
  /**
   * Remove a specific cached prediction
   */
  private async removeCachedPrediction(imageHash: string): Promise<void> {
    try {
      const cacheData = await AsyncStorage.getItem(this.CACHE_KEY);
      if (!cacheData) return;
      
      let cache: CachedPrediction[] = JSON.parse(cacheData);
      cache = cache.filter(item => item.imageHash !== imageHash);
      
      await AsyncStorage.setItem(this.CACHE_KEY, JSON.stringify(cache));
      
    } catch (error) {
      console.error('Error removing cached prediction:', error);
    }
  }
  
  /**
   * Clear all cached predictions
   */
  async clearCache(): Promise<void> {
    try {
      await AsyncStorage.removeItem(this.CACHE_KEY);
    } catch (error) {
      console.error('Error clearing cache:', error);
    }
  }
  
  /**
   * Get cache statistics
   */
  async getCacheStats(): Promise<{ size: number; oldestEntry: number | null }> {
    try {
      const cacheData = await AsyncStorage.getItem(this.CACHE_KEY);
      if (!cacheData) return { size: 0, oldestEntry: null };
      
      const cache: CachedPrediction[] = JSON.parse(cacheData);
      const oldestEntry = cache.length > 0 
        ? Math.min(...cache.map(item => item.timestamp))
        : null;
      
      return {
        size: cache.length,
        oldestEntry,
      };
      
    } catch (error) {
      console.error('Error getting cache stats:', error);
      return { size: 0, oldestEntry: null };
    }
  }
}

export const cacheService = new CacheService();
```

## Deployment Strategy

### 1. Model Deployment Options

#### Option A: Cloud API Deployment

```yaml
# docker-compose.yml for cloud deployment
version: '3.8'

services:
  ml-api:
    build:
      context: .
      dockerfile: Dockerfile.ml-api
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/best_model.h5
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./models:/app/models
    depends_on:
      - redis
    restart: unless-stopped
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - ml-api
    restart: unless-stopped
```

#### Option B: Edge Deployment (TensorFlow Lite)

```typescript
// services/EdgeMLService.ts
/**
 * Service for running TensorFlow Lite models on device
 */

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';

class EdgeMLService {
  private model: tf.LayersModel | null = null;
  private isModelLoaded = false;
  
  async loadModel(modelUrl: string): Promise<void> {
    try {
      console.log('Loading TensorFlow.js model...');
      
      // Initialize TensorFlow.js
      await tf.ready();
      
      // Load the model
      this.model = await tf.loadLayersModel(modelUrl);
      this.isModelLoaded = true;
      
      console.log('Model loaded successfully');
      console.log('Input shape:', this.model.inputs[0].shape);
      
    } catch (error) {
      console.error('Failed to load model:', error);
      throw new Error(`Model loading failed: ${error.message}`);
    }
  }
  
  async predict(imageUri: string): Promise<PredictionResult> {
    if (!this.isModelLoaded || !this.model) {
      throw new Error('Model not loaded');
    }
    
    try {
      // Load and preprocess image
      const imageTensor = await this.preprocessImage(imageUri);
      
      // Make prediction
      const prediction = this.model.predict(imageTensor) as tf.Tensor;
      const probabilities = await prediction.data();
      
      // Clean up tensors
      imageTensor.dispose();
      prediction.dispose();
      
      // Process results
      const benignProb = probabilities[0];
      const malignantProb = probabilities[1];
      const predictedClass = malignantProb > benignProb ? 'Malignant' : 'Benign';
      const confidence = Math.max(benignProb, malignantProb);
      
      return {
        predicted_class: predictedClass,
        confidence,
        probabilities: {
          benign: benignProb,
          malignant: malignantProb,
        },
        risk_level: this.getRiskLevel(confidence, predictedClass === 'Malignant' ? 1 : 0),
      };
      
    } catch (error) {
      console.error('Prediction error:', error);
      throw new Error(`Prediction failed: ${error.message}`);
    }
  }
  
  private async preprocessImage(imageUri: string): Promise<tf.Tensor> {
    // Load image as tensor
    const response = await fetch(imageUri);
    const imageData = await response.arrayBuffer();
    const imageTensor = tf.node.decodeImage(new Uint8Array(imageData), 3);
    
    // Resize to model input size
    const resized = tf.image.resizeBilinear(imageTensor, [224, 224]);
    
    // Normalize pixel values
    const normalized = resized.div(255.0);
    
    // Add batch dimension
    const batched = normalized.expandDims(0);
    
    // Clean up intermediate tensors
    imageTensor.dispose();
    resized.dispose();
    normalized.dispose();
    
    return batched;
  }
  
  private getRiskLevel(confidence: number, predictedClass: number): string {
    if (predictedClass === 0) { // Benign
      if (confidence > 0.9) return 'low';
      if (confidence > 0.7) return 'low-medium';
      return 'medium';
    } else { // Malignant
      if (confidence > 0.8) return 'high';
      if (confidence > 0.6) return 'medium-high';
      return 'medium';
    }
  }
  
  isReady(): boolean {
    return this.isModelLoaded;
  }
  
  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
      this.isModelLoaded = false;
    }
  }
}

export const edgeMLService = new EdgeMLService();
```

### 2. Monitoring and Analytics

```python
# monitoring.py
"""
Model monitoring and analytics system
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ModelMonitor:
    def __init__(self, log_file: str = 'model_predictions.log'):
        self.log_file = log_file
        self.logger = self._setup_logger()
        self.predictions_buffer = []
        self.buffer_size = 1000
    
    def _setup_logger(self):
        logger = logging.getLogger('model_monitor')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def log_prediction(self, 
                      image_id: str,
                      prediction: Dict[str, Any],
                      user_id: str = None,
                      metadata: Dict[str, Any] = None):
        """
        Log a model prediction for monitoring
        
        Args:
            image_id: Unique identifier for the image
            prediction: Model prediction results
            user_id: User who made the request
            metadata: Additional metadata
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'image_id': image_id,
            'user_id': user_id,
            'prediction': prediction,
            'metadata': metadata or {}
        }
        
        self.logger.info(json.dumps(log_entry))
        self.predictions_buffer.append(log_entry)
        
        # Flush buffer if it's full
        if len(self.predictions_buffer) >= self.buffer_size:
            self._process_buffer()
    
    def log_feedback(self,
                    image_id: str,
                    actual_label: str,
                    predicted_label: str,
                    confidence: float,
                    user_feedback: str = None):
        """
        Log user feedback for model improvement
        
        Args:
            image_id: Unique identifier for the image
            actual_label: Actual/correct label
            predicted_label: Model's predicted label
            confidence: Model's confidence score
            user_feedback: Optional user feedback
        """
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'image_id': image_id,
            'actual_label': actual_label,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'user_feedback': user_feedback,
            'correct_prediction': actual_label == predicted_label
        }
        
        self.logger.info(f"FEEDBACK: {json.dumps(feedback_entry)}")
    
    def _process_buffer(self):
        """Process the predictions buffer for analytics"""
        if not self.predictions_buffer:
            return
        
        # Calculate basic statistics
        confidences = [p['prediction']['confidence'] for p in self.predictions_buffer]
        avg_confidence = np.mean(confidences)
        
        malignant_predictions = sum(
            1 for p in self.predictions_buffer 
            if p['prediction']['predicted_class'] == 'Malignant'
        )
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(self.predictions_buffer),
            'avg_confidence': avg_confidence,
            'malignant_rate': malignant_predictions / len(self.predictions_buffer),
            'confidence_distribution': {
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'std': float(np.std(confidences))
            }
        }
        
        self.logger.info(f"STATS: {json.dumps(stats)}")
        
        # Clear buffer
        self.predictions_buffer = []
    
    def generate_performance_report(self, feedback_file: str = None):
        """
        Generate a performance report based on logged feedback
        
        Args:
            feedback_file: Optional file containing feedback data
        """
        # This would analyze the logged feedback to generate performance metrics
        # Implementation depends on your specific logging and storage setup
        pass

# Global monitor instance
model_monitor = ModelMonitor()
```

This comprehensive guide provides you with everything needed to implement a production-ready ML system for skin cancer detection. The documentation covers:

1. **Complete data pipeline** from collection to preprocessing
2. **Multiple model architectures** with transfer learning options
3. **Robust training pipeline** with monitoring and evaluation
4. **Production-ready inference** with both cloud and edge deployment
5. **Performance optimization** and caching strategies
6. **Monitoring and analytics** for continuous improvement

Each section includes detailed code examples and best practices for building a scalable, maintainable ML system.