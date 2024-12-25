# Building Generative AI Applications with HuggingFace and Gradio

A collection of AI applications demonstrating various capabilities using HuggingFace's models and Gradio interfaces. This project includes Natural Language Processing (NER) and Computer Vision (Image Captioning) implementations, with both API and local inference options.

## Overview

This project showcases different AI capabilities:

1. **NLP Tasks with Simple Interface**
   
   Build interfaces for various NLP tasks including:
   - Text summarization using BART model
   - Named Entity Recognition (NER) using BERT
   - Token merging and entity highlighting
   - Simple text processing applications

2. **Image Captioning**
   - Generates natural language descriptions of images
   - Uses BLIP model from Salesforce
   - Handles both uploaded images and image URLs
   - Includes example images for testing

3. **Image Generation**
   - Generates images based on text prompts
   - Uses Stable Diffusion model
   - Interactive prompt input for image generation
   - Example prompts for testing

4. **Describe-and-Generate Game Functions**
   - `image_to_base64_str()`: Converts PIL images to base64 strings
   - `base64_to_pil()`: Converts base64 strings back to PIL images
   - `captioner()`: Generates captions for images
   - `generate()`: Creates images from text prompts
   - `caption_and_generate()`: Combined function for the describe-and-generate game

### utils.py
The `utils.py` file contains helper functions

All applications feature:
- HuggingFace Inference API integration
- Local model fallback options
- Interactive Gradio web interfaces
- Example inputs for testing

## Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Set up your HuggingFace API token:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with "read" access
   - Create a `.env` file in your project root:
```bash
HF_API_KEY="your_token_here"
```

## Project Structure

```
.
├── README.md
├── code/
│   ├── L1_NLP_tasks_with_a_simple_interface.ipynb
│   └── L2_Image_captioning_app.ipynb
│   └── L3_Image_generation_app.ipynb
│   └── L4_Describe_and_generate_game.ipynb
│   └── utils.py
├── data/
│   ├── christmas_dog.jpeg
│   ├── bird_flight.jpeg
│   └── cow.jpeg
└── .env
```

## Notebooks

The project includes several Jupyter notebooks demonstrating different AI capabilities:

1. `L1_NLP_tasks_with_a_simple_interface.ipynb`
   - Text summarization using BART model
   - Named Entity Recognition (NER) using BERT
   - Token merging and entity highlighting
   - Simple text processing applications

2. `L2_Image_captioning_app.ipynb`
   - Image captioning using BLIP model
   - Processes both uploaded images and image URLs
   - Interactive Gradio interface for image captioning
   - Example images included in data folder

3. `L3_Image_generation_app.ipynb`
   - Image generation using Stable Diffusion model
   - Interactive Gradio interface for image generation
   - Example prompts included in data folder

### 4. Describe and Generate Game
   - Upload or select an image
   - AI generates a description of the image
   - Uses the description to generate a new image
   - Compare original and AI-generated images
   - Chain multiple generations to see how images evolv

## Usage

Each notebook can be run independently and includes detailed instructions. The general workflow is:

1. Set up environment and API keys
2. Import required functions from utils.py
3. Run the notebook cells sequentially
4. Interact with the Gradio interfaces

### Basic usage with the API:

```python
import os
import requests

API_URL = os.environ['HF_API_NER_BASE']  # NER endpoint
text = "My name is Miqueas, I'm building MMD Solutions and I live in Spain"

# Define the payload
payload = {
    "inputs": text,
    "parameters": None
}

# Make the POST request to the API endpoint
response = requests.post(API_URL, json=payload)

# Check if the request was successful
if response.status_code == 200:
    output = response.json()
    print(output)
else:
    print(f"Request failed with status code {response.status_code}: {response.text}")
```

### Running the Gradio Interface:
```python
import os
import gradio as gr
from transformers import pipeline

# Initialize the NER pipeline
get_completion = pipeline("ner", model="dslim/bert-base-NER")

# Define the NER function
def ner(input):
    output = get_completion(input)
    return {"text": input, "entities": output}

# Close any existing Gradio interfaces
gr.close_all()

# Create and launch the Gradio interface
demo = gr.Interface(fn=ner, inputs="text", outputs="json")
demo.launch(share=True, server_port=int(os.environ.get('PORT1', 7860)))
```

### Architecture
- Primary: HuggingFace Inference API
- Fallback: Local transformer pipeline
- Model: dslim/bert-base-NER
- Interface: Gradio web UI

## Troubleshooting

Common issues and solutions:

1. **API Token Invalid**
   - Verify token starts with "hf_"
   - Check token permissions
   - Create a new token if needed

2. **Model Loading Issues**
   - Ensure sufficient RAM (minimum 4GB)
   - Check internet connection for initial download

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## ☕ Support Me

If you like my work, consider supporting my studies!

Your contributions will help cover fees and materials for my **Computer Science and Engineering studies at UoPeople** starting in September 2025.

Every little bit helps—you can donate from as little as $1.

<a href="https://ko-fi.com/miqueasmd"><img src="https://ko-fi.com/img/githubbutton_sm.svg" /></a>

## Acknowledgements

This project is inspired by the DeepLearning.AI courses. Please visit [DeepLearning.AI](https://www.deeplearning.ai/) for more information and resources.