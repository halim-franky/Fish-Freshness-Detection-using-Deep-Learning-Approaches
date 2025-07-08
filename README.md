🐟 Fish Freshness Detection using Deep Learning
This project investigates the use of deep learning architectures—VGG16, Vision Transformer (ViT), and a hybrid VGG16-ViT model—to automatically classify fish freshness based on eye image data. It addresses the real-world challenge of subjective and labor-intensive fish freshness assessment, offering a foundation for developing scalable, AI-driven quality control solutions in the seafood industry.

📘 Project Summary
Ensuring the freshness of fish is critical for public health, food safety, and minimizing economic losses in seafood logistics. Traditional inspection methods are often subjective and inconsistent. In this project, we employ computer vision and transfer learning techniques to automate freshness detection using eye images from The Freshness of Fish Eyes Dataset.

The classification task categorizes images into three classes:

Highly Fresh

Fresh

Not Fresh

Despite class imbalances and subtle visual differences, our models attempt to generalize across categories through data augmentation and fine-tuning.

🔬 Research Objectives
Develop a deep learning pipeline to classify fish freshness using fish eye images.

Compare performance across:

VGG16 (CNN-based)

Vision Transformer (ViT) (Transformer-based)

Hybrid VGG16-ViT model

Address dataset imbalance with augmentation.

Evaluate classification performance with accuracy, confusion matrix, and classification report.

🧠 Model Architectures
1. VGG16
A convolutional neural network with 16 layers, known for hierarchical feature extraction. Transfer learning with ImageNet weights is used.

2. Vision Transformer (ViT)
Processes image patches using self-attention mechanisms, capturing global image context and long-range dependencies.

3. Hybrid VGG16-ViT
Combines local feature extraction (VGG16) and global representation (ViT) through concatenated feature embeddings.

🔄 Project Pipeline
mermaid
Copy
Edit
graph TD
    A[Raw Dataset] --> B[Dataset Splitting (Train/Val/Test)]
    B --> C[Data Augmentation (Balancing Classes)]
    C --> D[Image Preprocessing (Resize 224x224, Normalize)]
    D --> E[Model Training]
    E --> F[VGG16 | ViT | VGG16+ViT]
    F --> G[Evaluation: Accuracy, Confusion Matrix, F1-Score]
    G --> H[Insights & Limitations]
📊 Performance Summary
Model	Test Accuracy	Notes
VGG16	56.49%	Strongest generalization on "Fresh" class
ViT	52.61%	Struggled with minority classes
Hybrid Model	54.35%	Biased toward "Not Fresh"; needs optimization

🧪 Challenges & Limitations
Class Imbalance: "Fresh" images dominate the dataset.

Visual Similarity: Eye features between freshness categories are often subtle.

Overfitting Risks: Especially during hybrid training.

ViT Complexity: Requires substantial computational power.

Transfer Learning Bias: Pre-trained weights from ImageNet may not be well-suited to fish eye domains.

🔧 Libraries & Installation
Install Requirements
bash
Copy
Edit
pip install tensorflow==2.15.0 matplotlib opencv-python seaborn vit-keras tensorflow_addons
Key Libraries
TensorFlow / Keras

vit-keras

OpenCV

matplotlib, seaborn

scikit-learn

📁 Dataset
Source: The Freshness of Fish Eyes Dataset

Classes: Highly Fresh, Fresh, Not Fresh

Preprocessing:

Resized to 224×224 pixels

Normalized pixel values (0–1)

Augmentation applied to underrepresented classes:

Rotation (30°), width/height shift (20%), shear (20%), zoom (20%), horizontal flip

🧪 Evaluation Metrics
Accuracy

Precision / Recall / F1-Score

Confusion Matrix

Evaluation was performed on the 10% held-out test set after fine-tuning.

📌 Future Directions
Integrate gill and skin features to improve classification.

Expand dataset diversity (species, lighting, angles).

Apply custom attention mechanisms for fine-grained classification.

Package as a mobile app for fish vendors and consumers.

📷 Sample Visualization
python
Copy
Edit
# Visualize sample predictions
display_images("train", category="Highly Fresh")

![image](https://github.com/user-attachments/assets/0758d22f-3647-49e5-8439-c06db2d29b83)
