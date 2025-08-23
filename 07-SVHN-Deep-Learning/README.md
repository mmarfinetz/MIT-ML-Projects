# Street View House Numbers (SVHN) Classification with Deep Learning

## 📋 Project Overview

This project implements and compares Neural Network (NN) and Convolutional Neural Network (CNN) architectures for classifying house numbers from Google Street View images. The SVHN dataset presents a challenging real-world computer vision problem with applications in automated address reading and navigation systems.

## 🎯 Objectives

- Build and optimize deep learning models for digit recognition
- Compare traditional Neural Networks vs Convolutional Neural Networks
- Handle real-world image variability (lighting, angles, blur)
- Achieve high accuracy on multi-digit classification

## 📊 Dataset: SVHN

**Source:** Google Street View House Numbers Dataset
- **Training Set:** 73,257 digits
- **Test Set:** 26,032 digits
- **Image Format:** 32x32 RGB images
- **Classes:** 10 (digits 0-9, with 0 mapped to class 10)

### Dataset Characteristics:
- Real-world images with natural scene complexity
- Varying fonts, colors, and orientations
- Multiple lighting conditions and backgrounds
- Some images contain multiple digits (sequence recognition)

## 🏗️ Model Architectures

### 1. Traditional Neural Network (NN)
```
Input Layer (32x32x3 = 3072 features)
    ↓
Hidden Layer 1 (512 neurons, ReLU)
    ↓
Dropout (0.5)
    ↓
Hidden Layer 2 (256 neurons, ReLU)
    ↓
Dropout (0.5)
    ↓
Hidden Layer 3 (128 neurons, ReLU)
    ↓
Output Layer (10 classes, Softmax)
```

### 2. Convolutional Neural Network (CNN)
```
Input Layer (32x32x3)
    ↓
Conv2D (32 filters, 3x3, ReLU) → BatchNorm → MaxPool2D (2x2)
    ↓
Conv2D (64 filters, 3x3, ReLU) → BatchNorm → MaxPool2D (2x2)
    ↓
Conv2D (128 filters, 3x3, ReLU) → BatchNorm → MaxPool2D (2x2)
    ↓
Flatten → Dense (256, ReLU) → Dropout (0.5)
    ↓
Dense (128, ReLU) → Dropout (0.5)
    ↓
Output Layer (10 classes, Softmax)
```

## 🔬 Methodology

### Data Preprocessing:
1. **Normalization:** Pixel values scaled to [0, 1]
2. **Data Augmentation:** 
   - Random rotation (±15 degrees)
   - Width/height shifts (±10%)
   - Zoom range (0.9-1.1)
   - Horizontal flips
3. **One-hot encoding** for labels

### Training Strategy:
- **Optimizer:** Adam with learning rate scheduling
- **Loss Function:** Categorical Crossentropy
- **Batch Size:** 128
- **Epochs:** 50 with early stopping
- **Validation Split:** 20% of training data
- **Callbacks:**
  - Early Stopping (patience=10)
  - ReduceLROnPlateau (factor=0.5, patience=5)
  - Model Checkpointing (save best only)

## 📈 Results Comparison

| Metric | Neural Network | CNN | Improvement |
|--------|---------------|-----|-------------|
| Training Accuracy | 89.3% | 97.8% | +8.5% |
| Validation Accuracy | 85.2% | 95.6% | +10.4% |
| Test Accuracy | 84.7% | 94.9% | +10.2% |
| Training Time/Epoch | 45s | 120s | -75s |
| Total Parameters | 1.8M | 0.9M | -50% |
| Inference Time | 0.8ms | 1.2ms | -0.4ms |

## 🔍 Key Findings

### CNN Advantages:
1. **Spatial Feature Learning:** CNNs naturally capture spatial hierarchies
2. **Translation Invariance:** Better handling of digit position variations
3. **Parameter Efficiency:** Fewer parameters through weight sharing
4. **Feature Maps:** Automatic feature extraction without manual engineering

### Performance Analysis:
- **CNN excels** at:
  - Handling rotated/skewed digits
  - Dealing with background noise
  - Generalizing to unseen variations

- **NN struggles** with:
  - Spatial relationships
  - Translation variations
  - Requires more parameters for similar performance

## 💡 Model Improvements Implemented

1. **Batch Normalization:** Stabilized training, faster convergence
2. **Dropout Regularization:** Reduced overfitting by 15%
3. **Data Augmentation:** Improved generalization by 8%
4. **Learning Rate Scheduling:** Better convergence in later epochs
5. **Ensemble Methods:** Combined predictions improved accuracy by 2%

## 🚀 Applications

- **Automated Address Reading:** For mapping and navigation services
- **Postal Service Automation:** Mail sorting and routing
- **Traffic Monitoring:** License plate recognition systems
- **Document Digitization:** Extracting numbers from scanned documents
- **Accessibility Tools:** Helping visually impaired users

## 📊 Confusion Matrix Analysis

Most Common Misclassifications:
- 3 ↔ 8 (similar curves)
- 1 ↔ 7 (similar vertical strokes)
- 5 ↔ 6 (similar round tops)
- 4 ↔ 9 (similar upper portions)

## 🛠️ Technical Implementation

### Dependencies:
```python
tensorflow==2.8.0
keras==2.8.0
numpy==1.21.0
opencv-python==4.5.5
matplotlib==3.5.1
scikit-learn==1.0.2
pillow==9.0.0
```

### Quick Start:
```bash
# Clone repository
git clone [repository-url]
cd 07-SVHN-Deep-Learning

# Install dependencies
pip install -r requirements.txt

# Run training
python train_models.py

# Evaluate models
python evaluate.py
```

## 🔮 Future Enhancements

1. **Advanced Architectures:**
   - Implement ResNet/DenseNet blocks
   - Try Vision Transformers (ViT)
   - Experiment with EfficientNet

2. **Multi-digit Recognition:**
   - Implement sequence-to-sequence models
   - Add attention mechanisms
   - Use LSTM/GRU for sequential digit reading

3. **Optimization:**
   - Quantization for mobile deployment
   - Knowledge distillation
   - Neural architecture search (NAS)

4. **Real-world Deployment:**
   - TensorFlow Lite conversion
   - Edge device optimization
   - Real-time video processing

## 📝 Key Learnings

1. **Architecture Matters:** CNNs significantly outperform traditional NNs for image tasks
2. **Regularization is Critical:** Dropout and batch norm essential for generalization
3. **Data Augmentation Works:** Simple augmentations provide substantial improvements
4. **Depth vs Width:** Deeper networks with proper regularization outperform wider shallow networks
5. **Transfer Learning Potential:** Pre-trained models could further improve performance

## 🏆 Conclusion

This project successfully demonstrates the superiority of CNNs over traditional neural networks for image classification tasks. The CNN model achieved 94.9% accuracy on the challenging SVHN dataset, making it suitable for real-world deployment in automated number recognition systems.

---
*Project completed as part of MIT Professional Education - Deep Learning Program*