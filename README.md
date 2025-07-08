# 🐟 Fish Freshness Detection using Deep Learning


This project investigates the use of deep learning architectures including VGG16, Vision Transformer (ViT), and a hybrid VGG16-ViT model. This project use these deep learning models to automatically classify fish freshness based on eye image data. It addresses the real-world challenge of subjective and labor-intensive fish freshness assessment, offering a foundation for developing scalable, AI-driven quality control solutions in the seafood industry.

# 📘 Project Summary


Ensuring the freshness of fish is critical for public health, food safety, and minimizing economic losses in seafood logistics. Traditional inspection methods are often subjective and inconsistent. In this project, we employ computer vision and transfer learning techniques to automate freshness detection using eye images from The Freshness of Fish Eyes Dataset.

The classification task categorizes images into three classes:
<ul>
<li>Highly Fresh</li>

<li>Fresh</li>

<li>Not Fresh</li>
</ul>

Despite class imbalances and subtle visual differences, our models attempt to generalize across categories through data augmentation and fine-tuning.

# 🔬 Research Objectives


Develop a deep learning pipeline to classify fish freshness using fish eye images.

Compare performance across:

<ul>
<li>VGG16 (CNN-based)</li>

<li>Vision Transformer (ViT) (Transformer-based)</li>

<li>Hybrid VGG16-ViT model</li>

</ul>

Address dataset imbalance with augmentation.

Evaluate classification performance with accuracy, confusion matrix, and classification report.

# 🧠 Model Architectures


1. VGG16
A convolutional neural network with 16 layers, known for hierarchical feature extraction. Transfer learning with ImageNet weights is used.

2. Vision Transformer (ViT)
Processes image patches using self-attention mechanisms, capturing global image context and long-range dependencies.

3. Hybrid VGG16-ViT
Combines local feature extraction (VGG16) and global representation (ViT) through concatenated feature embeddings.

# 🔄 Project Pipeline
Dataset --> Dataset Splitting (Train/Val/Test) --> Data Augmentation (Balancing Classes) Image Preprocessing (Resize 224x224, Normalize) --> Model Training (VGG16 | ViT | VGG16+ViT) --> Evaluation: Accuracy, Confusion Matrix, F1-Score --> Insights & Limitations

# 📊 Performance Summary


Model	Test Accuracy	Notes
VGG16	56.49%	Strongest generalization on "Fresh" class
ViT	52.61%	Struggled with minority classes
Hybrid Model	54.35%	Biased toward "Not Fresh"; needs optimization

![image](https://github.com/user-attachments/assets/0758d22f-3647-49e5-8439-c06db2d29b83)

# 🧪 Challenges & Limitations

<ul>
<li>Class Imbalance: "Fresh" images dominate the dataset.</li>

<li>Visual Similarity: Eye features between freshness categories are often subtle.</li>

<li>Overfitting Risks: Especially during hybrid training.</li>

<li>ViT Complexity: Requires substantial computational power.</li>

<li>Transfer Learning Bias: Pre-trained weights from ImageNet may not be well-suited to fish eye domains.</li>
</ul>

# 🔧 Libraries & Installation


Install Requirements:
```
pip install tensorflow==2.15.0 matplotlib opencv-python seaborn vit-keras tensorflow_addons
```

Key Libraries:
- TensorFlow / Keras
- vit-keras
- OpenCV
- matplotlib, seaborn
- scikit-learn

# 📁 Dataset


Source: The Freshness of Fish Eyes Dataset

Classes: Highly Fresh, Fresh, Not Fresh

Preprocessing:

1. Resized to 224×224 pixels

2. Normalized pixel values (0–1)

3. Augmentation applied to underrepresented classes:

4. Rotation (30°), width/height shift (20%), shear (20%), zoom (20%), horizontal flip

# 🧪 Evaluation Metrics

<ul>
<li>Accuracy</li>

<li>Precision / Recall / F1-Score</li>

<li>Confusion Matrix</li>

<li>Evaluation was performed on the 10% held-out test set after fine-tuning.</li>
</ul>

# 📌 Future Directions

<ul>
<li>Integrate gill and skin features to improve classification.</li>

<li>Expand dataset diversity (species, lighting, angles).</li>

<li>Apply custom attention mechanisms for fine-grained classification.</li>

<li>Package as a mobile app for fish vendors and consumers.</li>
</ul>


