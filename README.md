# Overview

Our project, WasteWizard, is an AI-powered consumer app that uses computer vision to help users upload waste images and receive guidance on proper disposal methods. With the US recycling rate at just 32% (EPA 2018), WasteWizard categorizes waste into 26 sub-categories, including recyclables, e-waste, and hazardous materials, reducing manual sorting and contamination. The app aims to improve recycling efficiency, reduce landfill overflow, and minimize environmental risks from improper waste disposal.

## Methodology

In our waste sorting AI application, we tackled a multiclass image classification task across 26 waste categories. After merging the data, we resized images, removed transparency, background noise, and duplicates, normalized pixel values, and split the data with an 80/20 ratio for training, testing, and validation. Initially, mislabeled images negatively affected performance, but removing them led to significant improvements in model efficacy.


