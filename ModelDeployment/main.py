# WasteWizard Model API endpoint implemented using FastAPI

from contextlib import asynccontextmanager
from fastapi import FastAPI
from typing import List
import urllib
import numpy as np
import torch
from fastapi.middleware.cors import CORSMiddleware

# import os
from PIL import Image
from skimage.transform import resize
import urllib.request
from urllib.parse import urlparse
from io import BytesIO
import rembg

# Load the PyTorch model
path = 'vit_model.pth'
vit_model = torch.load(path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model.to(device)
vit_model.eval()

# Lifespan Event to only load model once when FastAPI first launches
# Source: https://fastapi.tiangolo.com/advanced/events/
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["wastewizard"] = vit_model
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(lifespan=lifespan)

# Allow requests from all origins (replace "*" with your allowed origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# DATA PREPROCESSING SCRIPT FROM ADA
def resize_with_padding(image, target_height, target_width):
    """
    Resize image with padding to maintain aspect ratio.

    """
    # Calculate aspect ratio of original image
    original_height, original_width, _ = image.shape
    aspect_ratio = original_width / original_height
    
    # Resize image while preserving aspect ratio and fill with white pixels
    if aspect_ratio > target_width / target_height:
        # Image is wider, resize based on width
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    elif aspect_ratio < target_width / target_height:
        # Image is taller, resize based on height
        new_height = target_height
        new_width = int(aspect_ratio * target_height)
    else:
        # Image has the same aspect ratio as target
        new_height = target_height
        new_width = target_width
    
    resized_image = resize(image, (new_height, new_width), mode='constant') * 255  # Fill with white pixels
    
    # Pad to target dimensions with white pixels if necessary
    padded_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255  # Fill with white pixels
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    padded_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
    
    return padded_image.astype(np.uint8)

def remove_transparency(im, bg_colour=(255, 255, 255)):
    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg.convert('RGB')

    else:
        return im

def process_image(image_url, target_height, target_width):
    # Download the image from the URL
    with urllib.request.urlopen(image_url) as response:
        img_data = response.read()

# Open the image using PIL
    with Image.open(BytesIO(img_data)) as img:
        img = img.convert("RGB")
        # Remove transparency
        img = remove_transparency(img)
        img_array = np.array(img).astype(np.uint8)

        # Resize and pad the image
        processed_image = resize_with_padding(img_array, target_height, target_width)

        # Normalize the image
        processed_image = processed_image.astype(np.float32) / 255.0


    return processed_image
        
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict/")
async def predict_images(image_urls: List[str]):
    results = []
    labels = ['battery', 'beverage cans', 'cardboard', 'cigarette butt',
            'construction scrap', 'electrical cables', 'electronic chips',
            'glass', 'gloves', 'laptops', 'masks', 'medicines',
            'metal containers', 'news paper', 'paper',
            'paper_cups', 'plastic bags', 'plastic bottles',
            'plastic containers', 'plastic_cups', 'small appliances',
            'smartphones', 'spray cans', 'syringe', 'tetra pak', 'trash']
    general_recycling = {
        'battery': 'e-Waste Disposal',
        'beverage cans': 'Metal',
        'cardboard': 'Paper',
        'cigarette butt': 'General Trash',
        'construction scrap': 'Metal',
        'electrical cables': 'e-Waste Disposal',
        'electronic chips': 'e-Waste Disposal',
        'glass': 'Glass',
        'gloves': 'Hazardous Waste',
        'laptops': 'e-Waste Disposal',
        'masks': 'Hazardous Waste',
        'medicines': 'General Trash',
        'metal containers': 'Metal',
        'news paper': 'Paper',
        'paper': 'Paper',
        'paper_cups': 'Paper',
        'plastic bags': 'General Trash',
        'plastic bottles': 'Plastic',
        'plastic containers': 'Plastic',
        'plastic_cups': 'Plastic',
        'small appliances': 'e-Waste Disposal',
        'smartphones': 'e-Waste Disposal',
        'spray cans': 'Hazardous Waste',
        'syringe': 'Hazardous Waste',
        'tetra pak': 'Paper',
        'trash': 'Trash'
    }
    check_recyclable = {
        'battery': False,
        'beverage cans': True,
        'cardboard': True,
        'cigarette butt': False,
        'construction scrap': True,
        'electrical cables': False,
        'electronic chips': False,
        'glass': True,
        'gloves': False,
        'laptops': False,
        'masks': False,
        'medicines': False,
        'metal containers': True,
        'news paper': True,
        'paper': True,
        'paper_cups': True,
        'plastic bags': False,
        'plastic bottles': True,
        'plastic containers': True,
        'plastic_cups': True,
        'small appliances': False,
        'smartphones': False,
        'spray cans': False,
        'syringe': False,
        'tetra pak': True,
        'trash': False
    }

    try:
        for url in image_urls:
            # Load and preprocess the image from the provided URL
            image_array = process_image(url, 224, 224)
            image_tensor = torch.tensor(image_array, dtype=torch.float32)
#             image_tensor_normalized = image_tensor / 255.0
#             image_tensor_unsqueezed = image_tensor_normalized.unsqueeze(0)
            image_tensor_unsqueezed = image_tensor.unsqueeze(0)
            image_tensor_transposed = np.transpose(image_tensor_unsqueezed, (0, 3, 1, 2))

            # Perform prediction
            with torch.no_grad():
                input = image_tensor_transposed.to(device)
                output = ml_models["wastewizard"](input)
                logits = output.logits
                predicted_label = torch.argmax(logits, 1)
                softmax_probs = torch.softmax(logits, 1)
                softmax_prob = softmax_probs[0, predicted_label].cpu().numpy()

            # Set confidence category
            if softmax_prob <= 0.15:
                confidence_level = "low"
            elif 0.15 < softmax_prob < 0.30:
                confidence_level = "medium"
            else:
                confidence_level = "high"
            
            # Retrieve string for predicted label
            wastetype_specific = labels[predicted_label]

            # Retrieve general recycling category for predicted label
            wastetype_general = general_recycling[wastetype_specific]

            # Specific Subcategory recommendation
            wastetype_suggested = "Trash" if confidence_level == 'low' else wastetype_specific

            # Set is_recyclable
            is_recyclable = check_recyclable[wastetype_specific]
            if wastetype_specific != "Trash" and confidence_level == 'low':
                is_recyclable = False

            # # Get image filename
            # parsed_url = urlparse(url)
            # filename = os.path.basename(parsed_url.path)

            # Build response
            results.append({"image_url": url,
                            # "filename": filename,
                            "wastetype_specific": wastetype_specific,
                            "wastetype_suggested": wastetype_suggested,
                            "wastetype_general": wastetype_general,
                            "is_recyclable": is_recyclable,
                            "confidence_level": confidence_level,
                            "softmax_probability": np.round(float(softmax_prob), 6),
                            })

        return {"predictions": results}

    except Exception as e:
        return {"error": str(e)}

