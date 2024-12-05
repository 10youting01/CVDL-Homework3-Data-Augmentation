# Homework3-Data-Augmentation
# README: Image Captioning and Augmentation Pipeline

## Overview

This repository contains a pipeline for generating captions for images, augmenting those images using Stable Diffusion, and creating enhanced visualizations. It employs state-of-the-art models like **BLIP2** for captioning and **Stable Diffusion** (including GLIGEN) for image generation and manipulation.

The pipeline includes two main stages:
1. **Image Captioning and Prompt Generation**: Captions and prompts are generated based on the content and metadata of images.
2. **Image Augmentation**: Using the generated captions and prompts, the images are enhanced or manipulated to create augmented visual outputs.

## Requirements

### Hardware
- A machine with a CUDA-enabled GPU is recommended for optimal performance.

### Software and Libraries
- Python 3.7+
- Required Python Libraries:
  - `transformers`
  - `torch`
  - `diffusers`
  - `Pillow`
  - `json`
  - `os`
- Pre-trained models:
  - `Salesforce/blip2-opt-2.7b` or `Salesforce/blip2-opt-6.7b` for captioning
  - `runwayml/stable-diffusion-v1-5` for image generation
  - `masterful/gligen-1-4-inpainting-text-box` for inpainting with bounding boxes

### Dataset
- An image folder (`images`) containing `.png`, `.jpg`, or `.jpeg` files.
- A JSON file (`label.json`) with metadata for each image, including:
  - `image`: Filename of the image.
  - `labels`: List of labels.
  - `height` and `width`: Image dimensions.
  - `bboxes`: Bounding box information.

## Pipeline Steps

### Stage 1: Image Captioning and Prompt Generation
1. **Caption Generation**:
   - Load BLIP2 (2.7B or 6.7B model) and generate captions for each image.
   - The captions describe the image content in natural language.
2. **Prompt Enrichment**:
   - Combine the generated caption with metadata such as labels, dimensions, and bounding boxes to create enriched prompts.

3. **Output**:
   - Save the generated data (captions, prompts, and metadata) to a JSON file (`image_data_blip2-opt-2.7b.json` or `image_data_blip2-opt-6.7b.json`).

### Stage 2: Image Augmentation
#### Using Stable Diffusion
1. **Augmentation with Text Prompts**:
   - Use enriched prompts from Stage 1 to generate augmented images.
   - Save the augmented images in the `augmented_images_suffix` or `augmented_images_text` folder.

2. **Augmentation with GLIGEN**:
   - Use GLIGEN to generate images with bounding boxes and inpainting based on the provided metadata and enriched prompts.
   - Save the output in the `augmented_images_gligen` folder.

3. **Resizing Images**:
   - Resize all generated images to a standard size (512x512 pixels) for further evaluation or usage.

4. **Evaluation**:
   - Use FID (Fréchet Inception Distance) to measure the quality of the generated images compared to the original dataset.

### Final Visualization
- Generate a final batch of 200 augmented images using GLIGEN.
- Save the results in the `generation_200` folder and resize them to `generation`.

## File Structure

```
├── images/                     # Input images
├── label.json                  # Metadata for input images
├── augmented_images_suffix/    # Images augmented using enriched prompts
├── augmented_images_text/      # Images augmented using captions
├── augmented_images_gligen/    # Images augmented using GLIGEN
├── generation_200/             # Final batch of 200 augmented images
├── generation/                 # Resized images from generation_200
├── image_data_blip2-opt-2.7b.json  # Generated captions and metadata
├── image_data_blip2-opt-6.7b.json  # Generated captions and metadata
└── README.md                   # This file
```

## How to Use

### 1. Prepare the Environment
Install the required Python libraries:
```bash
pip install torch transformers diffusers Pillow
```

### 2. Prepare the Dataset
- Place your images in the `images/` folder.
- Ensure a `label.json` file exists with metadata for each image.

### 3. Run the Code
- Run Stage 1 (caption generation):
  ```python
  python stage1_caption_generation.py
  ```
- Run Stage 2 (image augmentation):
  ```python
  python stage2_image_augmentation.py
  ```

### 4. Evaluate Results
- Compare generated images with original images using FID:
  ```bash
  python -m pytorch_fid resized_images_suffix resized_images
  ```

## Notes
- The pipeline supports multiple configurations (e.g., BLIP2 2.7B or 6.7B).
- Ensure adequate GPU memory for processing large models like BLIP2 and GLIGEN.
- FID scores provide a quantitative measure of image quality.

## Contact
For questions or feedback, please feel free to reach out!
