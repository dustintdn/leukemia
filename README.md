# Leukemia Blood Cell Classification

A deep learning project exploring neural network architectures for classifying microscopic blood cell images to distinguish healthy cells from leukemia blasts.

## Motivation

Acute Lymphoblastic Leukemia (ALL) diagnosis still relies heavily on qualitative evaluation by oncologists examining blood smears. Identifying leukemic blasts remains challenging, and patients often don't seek testing until symptoms appear. 

This project explores whether deep learning could help flag potential leukemia from routine blood tests, potentially enabling earlier detection—especially important for children where ALL is most common. While this uses a publicly available dataset rather than clinical data, it serves as a proof-of-concept for automated blood cell classification.

## Dataset

**Source:** [Kaggle - Leukemia Classification](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification)

The C-NMC dataset contains 15,135 microscopic stained blood cell images from 118 patients with two classes:
- **ALL** - Leukemia Blast cells
- **HEM** - Normal (Healthy) cells

For this exploration, I sampled 1,000 images for training, validation, and testing.

```
data/
└── C-NMC_Leukemia/
    └── training_data/
        ├── fold_0/
        │   ├── all/
        │   └── hem/
        ├── fold_1/
        │   ├── all/
        │   └── hem/
        └── fold_2/
            ├── all/
            └── hem/
```

## Model Experiments & Results

I tested five different architectures to understand tradeoffs between complexity, accuracy, and training time:

| Model | Architecture | Epochs | Accuracy | Notes |
|-------|-------------|--------|----------|-------|
| VGG16 | Transfer Learning | 5 | 50.0% | Failed to learn |
| CNN-32 | 4 layers, 32 filters | 20 | 79.3% | Overtrained |
| CNN-16 | 4 layers, 16 filters | 20 | 77.9% | Faster, similar results |
| CNN-Dropout | 3 layers + dropout | - | 74.5% | Regularization hurt performance |
| EfficientNetB3 | Transfer Learning | - | **98.0%** | Best accuracy, high loss |

### Detailed Analysis

**VGG16 (Transfer Learning)**  
The model converged at exactly 50% accuracy—essentially random guessing for binary classification. The pre-trained weights didn't transfer well to this domain without fine-tuning.

**CNN with 32 Filters**  
Built a custom CNN with filter size 32, alternating between kernel sizes 3 and 1. Accuracy reached ~76% by epoch 1, ~77% by epoch 3, and didn't break 79% until epoch 17. In hindsight, 5-10 epochs would have been sufficient—the extra training risked overfitting without meaningful gains.

**CNN with 16 Filters**  
Same architecture with smaller filters. Ran faster but performed slightly worse at 76.6% with 30 epochs. Learning curves suggested ~20 epochs was optimal, which achieved 77.9%.

**CNN with Dropout**  
Three convolutional layers (32→16→8 filters) with dropout (0.2) and learning rate 0.0001. The regularization actually hurt performance here, likely because the model was already underfitting.

**EfficientNetB3 (Transfer Learning)**  
By far the best performer at 98% accuracy. EfficientNet's compound scaling (width, depth, resolution) adapts well to the input images while staying lightweight. However, the high loss suggests overfitting—more regularization or data augmentation would help in production.

## Key Takeaways

1. **Accuracy vs. Compute Time** — Complex models can achieve better accuracy but at significant computational cost. For this dataset, a simple CNN achieved 79% accuracy in minutes while EfficientNet needed longer but hit 98%.

2. **Transfer Learning is Tricky** — VGG16 failed completely while EfficientNetB3 excelled. Architecture matters more than just using pre-trained weights.

3. **More Epochs ≠ Better** — Most of my models showed diminishing returns after 5-10 epochs. Monitoring validation loss is crucial.

4. **Regularization Isn't Always Helpful** — Dropout hurt performance on the smaller CNN, suggesting the model was already struggling to fit the data.

## Project Structure

```
leukemia-classification/
├── README.md
├── requirements.txt
├── notebooks/
│   └── model_exploration.ipynb    # All experiments and analysis
├── src/
│   ├── data_loader.py             # Data loading utilities
│   └── models.py                  # Model architectures
└── models/                        # Saved model weights (gitignored)
```

## Setup

```bash
# Clone and setup
git clone https://github.com/dustintdn/leukemia-classification.git
cd leukemia-classification

# Create environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Download data from Kaggle and extract to data/
```

## Usage

Open `notebooks/model_exploration.ipynb` to see all experiments, or use the modules directly:

```python
from src.data_loader import load_data
from src.models import build_cnn, build_efficientnet

# Load data
train_gen, val_gen, test_gen = load_data('data/C-NMC_Leukemia')

# Train a model
model = build_cnn(filters=32)
model.fit(train_gen, validation_data=val_gen, epochs=10)
```

## Future Improvements

- Data augmentation to reduce EfficientNet overfitting
- Class balancing / weighted loss functions
- Grad-CAM visualizations to interpret predictions
- Hyperparameter tuning with cross-validation
- Test on held-out patient data (not just held-out images)

## License

This project is for educational purposes. The dataset is from [The Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/).
