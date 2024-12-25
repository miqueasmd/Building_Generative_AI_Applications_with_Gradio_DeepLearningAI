import os
import base64
import random
import requests
import json
import time
from IPython.display import display, HTML
from PIL import Image
import io
import gradio as gr
from dotenv import load_dotenv, find_dotenv
from config import STABLE_DIFFUSION_PARAMS, DEFAULT_PARAMS

# Load environment variables
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']

# Replace the internal endpoint with a public HuggingFace API endpoint
ENDPOINT_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"


#Summarization endpoint
def get_completion_summary(inputs, parameters=None):
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    
    data = {"inputs": inputs}
    if parameters is not None:
        data.update({"parameters": parameters})
        
    response = requests.post(
        ENDPOINT_URL,
        headers=headers,
        json=data
    )
    return response.json()

def get_completion_api(text): 
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    API_URL = "https://api-inference.huggingface.co/models/dslim/bert-base-NER"
    data = { "inputs": text }
    
    # Debug info (don't share the full token)
    print(f"Using API URL: {API_URL}")
    print(f"Token starts with: {hf_api_key[:5]}...")
    
    response = requests.request("POST",
                            API_URL, 
                            headers=headers,
                            data=json.dumps(data))
    
    # Debug response
    if response.status_code != 200:
        print(f"API Response Status: {response.status_code}")
        print(f"API Response: {response.text}")
        
    return json.loads(response.content.decode("utf-8"))

def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # If current token continues the entity of the last one, merge them
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Otherwise, add the token to the list
            merged_tokens.append(token)
    return merged_tokens

def get_image_caption_from_url(image_url):
    """
    Get image caption using HuggingFace's BLIP model
    """
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
    
    # For URLs, we can send them directly
    data = { "inputs": image_url }
    
    response = requests.post(API_URL, headers=headers, json=data)
    return response.json()

def image_to_base64_str(pil_image):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()
    return str(base64.b64encode(byte_arr).decode('utf-8'))

def captioner(image):
    base64_image = image_to_base64_str(image)
    result = get_image_caption_from_url(base64_image)
    
    # Handle the API response format
    if isinstance(result, list):
        return result[0]['generated_text']
    elif isinstance(result, dict) and 'generated_text' in result:
        return result['generated_text']
    else:
        print("Unexpected API response format:", result)  # For debugging
        return "Could not generate caption"

def get_completion(prompt, params=None, preset="balanced"):
    """
    Generate image from text prompt using Stable Diffusion
    """
    url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    
    # If no params provided, use preset
    if params is None:
        params = DEFAULT_PARAMS[preset].copy()
        params["seed"] = random.randint(1, 1000000)
    
    payload = {
        "inputs": prompt,
        **params
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return base64.b64encode(response.content).decode('utf-8')
    else:
        print(f"Error: {response.status_code}")
        return None

def get_next_filename(base_name, directory):
    """Find the next available filename in sequence"""
    counter = 1
    while True:
        filename = f"{base_name}_{counter}.jpg"
        if not os.path.exists(os.path.join(directory, filename)):
            return filename
        counter += 1

def generate_and_save_images(prompt, num_images, data_path, hf_api_key, base_name="generated_image", delay=3):
    """Generate multiple images and save them with sequential filenames"""
    for _ in range(num_images):
        # Get next available filename
        filename = get_next_filename(base_name, data_path)
        
        # Generate new image
        result = get_completion(prompt, hf_api_key)
        if result:
            # Display the image
            display(HTML(f'<img src="data:image/jpeg;base64,{result}" />'))
            
            # Save the image
            image_data = base64.b64decode(result)
            with open(os.path.join(data_path, filename), "wb") as f:
                f.write(image_data)
            
            print(f"Saved as {filename}")
            
            # Wait to avoid rate limiting
            time.sleep(delay)

def get_image_caption(image_path, hf_api_key):
    """Get caption for an image using Hugging Face API"""
    url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        
    response = requests.post(url, headers=headers, data=image_bytes)
    return response.json()

def encode_image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def display_image_with_caption(image_path, caption=None):
    """Display image with optional caption in notebook"""
    img = Image.open(image_path)
    display(img)
    if caption:
        print(f"Caption: {caption}")

def process_image_batch(directory, hf_api_key, extension=".jpg"):
    """Process all images in a directory and get their captions"""
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            image_path = os.path.join(directory, filename)
            caption = get_image_caption(image_path, hf_api_key)
            results.append({
                'filename': filename,
                'path': image_path,
                'caption': caption
            })
    return results

#A helper function to convert the PIL image to base64
#so you can send it to the API

def base64_to_pil(img_base64):
    """Convert base64 string to PIL Image"""
    if img_base64 is None:
        raise ValueError("No image data received from API")
    try:
        base64_decoded = base64.b64decode(img_base64)
        byte_stream = io.BytesIO(base64_decoded)
        pil_image = Image.open(byte_stream)
        return pil_image
    except Exception as e:
        print(f"Error converting base64 to PIL: {e}")
        return None

def generate(prompt, negative_prompt="", steps=25, guidance=7, width=512, height=512):
    """Generate image for Gradio interface
    
    Args:
        prompt (str): The main prompt for image generation
        negative_prompt (str, optional): Things to avoid in the image. Defaults to "".
        steps (int, optional): Number of inference steps. Defaults to 25.
        guidance (float, optional): Guidance scale. Defaults to 7.
        width (int, optional): Image width. Defaults to 512.
        height (int, optional): Image height. Defaults to 512.
    
    Returns:
        PIL.Image: Generated image
    """
    # Convert numeric inputs to correct type (Gradio might send them as strings)
    steps = int(steps)
    guidance = float(guidance)
    width = int(width)
    height = int(height)
    
    params = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height,
        "seed": random.randint(1, 1000000)
    }
    
    try:
        output = get_completion(prompt, params=params)
        if output is None:
            print("Failed to generate image")
            return None
        
        return base64_to_pil(output)
    except Exception as e:
        print(f"Error in generate function: {e}")
        return None
    

