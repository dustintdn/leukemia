# Image Classification of Leukemia Red Blood Cells using Neural Networks

## Scope/Research Question

In this repository, I will create a model for image classification of blood cells to distinguish healthy blood cells and leukemia-infected blood cells. The goal is to automate leukemia detection and validation in cancer diagnosis using deep learning. The reason why this task is important and still needs improvement is because it is still difficult to identify leukemic blasts caused by Acute Lymphoblastic Leukemia (ALL). The purpose of this tool is to assist oncologists with their often qualitative evaluation of cancer disease.

## Data Description

The project dataset can be found here: https://www.kaggle.com/datasets/andrewmvd/leukemia-classification. The dataset contains 15,135 microscopic stained blood cell images from 118 patients. They are labeled as either Normal Cell (HEM) or Leukemia Blast (ALL). Of data provided, I will be sampling 1,000 images for training, validation, and testing. 

## Why utilize deep learning?

Deep learning systems would be a good methodological choice for this problem due to the current qualitative evaluation in cancer diagnosis in oncology. Patients are often not 'looking' to see if they have cancer and will not go to doctor until they are experiencing symptoms. If there were to be a way to automatically flag potential leukemia blasts from routine check-up lab tests (blood samples) by incorporating deep learning systems, it would help a lot of children identify ALL during early onset. Although this is a trivial dataset found off of kaggle, the development of the DL blood cell classification model could contribute to the improvement to problems identified in question #1. 

## Network frameworks

1. VGG16

2. CNN using 32 filters with 4 layers

3. CNN with smaller filters than Model #2

4. Another CNN; simplified with dropout function

5. EfficientNetB3

## Results / Critique Commentary

1. VGG16

The first model was a transfer learning model utilizing the VGG16 network architecture. The model accuracy converged at 0.5000 with 5 epochs. A binary classification model with 50% accuracy has very poor performance because there is an equal chance of guessing correct when there are only two categories. The loss was not sufficient for this model and made it incapable of performing.

2. CNN using 32 filters with 4 layers

This model convolved the image data using a filter size of 32 and alternated between kernal size 3 and 1 in each layer. The accuracy for this model was 0.7925. The performance of this model was an improvement from my VGG16 model, but the model used 20 epochs which could be causing overfitting. In the first epoch, model accuracy was ~0.76, in the third epoch accuracy was ~ 0.77, by the 10th epochs accuracy was ~0.78, and we did not break 0.79 accuracy until the 17th epoch. In terms of cost/time effectiveness, 5-10 epochs would've been more than enough to obtain similar accuracy and avoid the risk of overfitting. 

3. CNN with smaller filters than Model #2

Using the same network architecture as the previous model, this time with size 16 filters. This model performed worse but obvious ran faster epochs. With 30 epochs, the model accuracy was 0.7663. Utilizing the graphic showing the learning curve of the training/validation accuracy, ~20 epochs would've been the number for model fitting. Once reran with 20 epochs, the accuracy was 0.7794.

4. Another CNN; simplified with dropout 

In this model, we use three convolutional layers of 32, 16, and 8 filters respectively. I set the learning_rate=.0001 and a dropout function (0.2) is utilized in the second layer. This model's accuracy was 0.7451.

5. EfficientNetB3

In this transfer learning model utilizing EfficientNetB3, the accuracy was 0.9804. This model had extremely high loss and is prone to overfitting.

# More Information

## Directory Structure

```bash
C-NMC_Leukemia
|__ training_data
    |______ fold_0
        |______ all: []
        |______ hem: []
    |______ fold_1
        |______ all: []
        |______ hem: []
    |______ fold_2
        |______ all: []
        |______ hem: []
```
## Personal Takeaways

This project example has been insightful in the contruction of deep learning computer vision modeling. I found the biggest trade offs that were under consideraition in model decision-making and optimization was accuracy vs. time. Although complex models can produce more accurate results, the complexity of the model is computationally taxing and can cause extremely long wait times. This friction may obviously be resolved through upgraded hardware or cloud computing, but that results in higher operating costs for users. This project served as an opportunity test-drive various network architectures and features of neural networks to compare output, construction, and efficiency. Many of the models have infinite ways to be approached for improvmenets.
