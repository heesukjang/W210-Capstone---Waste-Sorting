# Overview

Our project, WasteWizard, is an AI-powered consumer app that uses computer vision to help users upload waste images and receive guidance on proper disposal methods. With the US recycling rate at just 32% (EPA 2018), WasteWizard categorizes waste into 26 sub-categories, including recyclables, e-waste, and hazardous materials, reducing manual sorting and contamination. The app aims to improve recycling efficiency, reduce landfill overflow, and minimize environmental risks from improper waste disposal.

In our waste sorting AI application, we tackled a multiclass image classification task across 26 waste categories. After merging the data, we resized images, removed transparency, background noise, and duplicates, normalized pixel values, and split the data with an 80/20 ratio for training, testing, and validation. Initially, mislabeled images negatively affected performance, but removing them led to significant improvements (17%) in model efficacy.

In the backend, we used the Vision Transformer (ViT) as our best-performing model for waste image classification, achieving a 90% macro F1 score, 91% precision, and 89% recall. The ViT classifies items as recyclable or non-recyclable, identifies waste type, and provides a certainty score based on softmax logit probabilities. In the ML pipeline, I focused on data wrangling, cleansing, exploratory data analysis (EDA), and developed additional models like CNNs, using Transfer Learning techniques (e.g., ResNet50, VGG19) with hyperparameter tuning techniques such as Optuna and Random Search.

For the UI frontend, I led the model deployment and developed a website using AWS Amplify, AppSync, S3, DynamoDB, and EC2, enabling users to upload images and view the model's results.

The business case is described in more detail in:
* final presentation
* and final demo video, but the results of this analysis have significant benefits for both airlines and passengers.

