# Project Overview
###### <i>UC Berkeley MIDS `2024 Spring Capstone` Group Project</i>

**WasteWizard** is an AI-powered consumer application that utilizes computer vision to help users upload images of waste and receive tailored guidance on proper disposal methods. With the US recycling rate at only 32% (EPA 2018), WasteWizard categorizes waste into 26 sub-categories, including recyclables, e-waste, and hazardous materials, minimizing the need for manual sorting and reducing contamination. The app aims to enhance recycling efficiency, reduce landfill overflow, and mitigate environmental risks associated with improper waste disposal.

In our AI waste sorting solution, we addressed a complex multiclass image classification task covering 26 waste categories. After merging the data, we performed extensive preprocessing, including resizing images, removing transparency and background noise, eliminating duplicates, normalizing pixel values, and splitting the dataset with an 80/20 ratio for training, testing, and validation. Initially, mislabeled images hindered model performance, but their removal improved efficacy by 17%.

On the backend, we leveraged the **Vision Transformer (ViT)** as our top-performing model, achieving a macro F1 score of 90%, with 91% precision and 89% recall. ViT efficiently classifies items as recyclable or non-recyclable, identifies waste types, and assigns certainty scores using softmax logit probabilities. Additionally, I focused on data wrangling, cleansing, and exploratory data analysis (EDA), building advanced models like CNNs and employing Transfer Learning techniques (e.g., ResNet50, VGG19) with hyperparameter tuning strategies like Optuna and Random Search.

For the frontend, I led the model deployment and developed a user-friendly website using **AWS Amplify, AppSync, S3, DynamoDB, and EC2**, allowing users to upload waste images and review classification results in real-time.

<h3>Tehnologies Used:</h3>
<li><strong>Backend:</strong> Python, FastAPI, Torch, Transformers, Sklearn, Keras, TensorFlow, Optuna, Pandas, Numpy</li>
<li><strong>Frontend:</strong> AWS Amplify, AppSync, S3, DynamoDB, EC2</li>

<h3>Additional Resources:</h3>
<ul>
 <li><a href="https://www.ischool.berkeley.edu/projects/2024/wastewizard" target="_blank">MIDS Capstone Project Spring 2014</a></li>
 <li><a href="https://www.youtube.com/watch?v=cUeJPhyFcGI&t=1s" target="_blank">Final Demo Video</a></li>
 <li><a href="https://github.com/heesukjang/WasteWizardWithComputerVision/blob/main/Final%20Presentation.pdf" target="_blank">Final Presentation</a></li>
 <li><a href="https://github.com/efficient-waste-sorting-org/ui-capstone-efficient-waste-sorting-2024/tree/main" target="_blank">UI Code</a></li>
</ul>


 

