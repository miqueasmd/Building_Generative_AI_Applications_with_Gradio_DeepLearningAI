# Named Entity Recognition (NER) with HuggingFace

This project implements a Named Entity Recognition system using HuggingFace's transformers library. It can work with both the HuggingFace Inference API and a local pipeline, providing a fallback mechanism for reliability.

## Features

- Uses the `dslim/bert-base-NER` model for entity recognition
- Supports both API and local inference
- Graceful fallback from API to local pipeline
- Interactive web interface using Gradio
- Identifies entities like:
  - Persons (PER)
  - Organizations (ORG)
  - Locations (LOC)

## Setup

1. Install required packages: requirements.txt


2. Set up your HuggingFace API token:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with "read" access
   - Create a `.env` file in your project root:


## Usage

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