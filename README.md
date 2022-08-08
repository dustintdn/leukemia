## Image Classification of Leukemia Red Blood Cells using Neural Networks

# Scope/Research Question

In this repository, I will create a model for image classification of blood cells to distinguish healthy blood cells and leukemia-infected blood cells. The goal is to automate leukemia detection and validation in cancer diagnosis using deep learning. The reason why this task is important and still needs improvement is because it is still difficult to identify leukemic blasts caused by Acute Lymphoblastic Leukemia (ALL). The purpose of this tool is to assist oncologists with their often qualitative evaluation of cancer disease.

# Data Description

The project dataset can be found here: https://www.kaggle.com/datasets/andrewmvd/leukemia-classification. The dataset contains 15,135 microscopic stained blood cell images from 118 patients. They are labeled as either Normal Cell (HEM) or Leukemia Blast (ALL). Of data provided, I will be sampling 1,000 images for training, validation, and testing. 

# Why utilize deep learning?

Deep learning systems would be a good methodological choice for this problem due to the current qualitative evaluation in cancer diagnosis in oncology. Patients are often not 'looking' to see if they have cancer and will not go to doctor until they are experiencing symptoms. If there were to be a way to automatically flag potential leukemia blasts from routine check-up lab tests (blood samples) by incorporating deep learning systems, it would help a lot of children identify ALL during early onset. Although this is a trivial dataset found off of kaggle, the development of the DL blood cell classification model could contribute to the improvement to problems identified in question #1. 

# Network frameworks

1. VGG16

2. CNN using 32 filters with 4 layers

3. CNN with smaller filters than Model #2

4. Another CNN; simplified with dropout function

5. EfficientNetB3

# Results

1. VGG16

2. CNN using 32 filters with 4 layers

3. CNN with smaller filters than Model #2

4. Another CNN; simplified with dropout function

5. EfficientNetB3

## More Information

# Directory Structure

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
        
# Personal Takeaways

This project example has been insightful in the contruction of deep learning computer vision modeling. I found the biggest trade offs that were under consideraition in model decision-making and optimization was accuracy vs. time. Although complex models can produce more accurate results, the complexity of the model is computationally taxing and can cause extremely long wait times. This friction may obviously be resolved through upgraded hardware or cloud computing, but that results in higher operating costs for users.
