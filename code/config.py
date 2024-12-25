# Stable Diffusion parameters and their explanations
STABLE_DIFFUSION_PARAMS = {
    "seed": {
        "range": "Any integer",
        "description": "Controls randomness. Same seed + prompt = same image",
        "default": None  # Will be randomly generated if not provided
    },
    "negative_prompt": {
        "range": "Any text",
        "description": "What you don't want in the image",
        "examples": ["blurry, bad quality, ugly, deformed"],
        "default": None
    },
    "num_inference_steps": {
        "range": "1-100 (typical)",
        "description": "Number of denoising steps",
        "ranges": {
            "fast": "10-20: Fast but rough",
            "balanced": "20-30: Good balance",
            "quality": "50+: High quality but slower",
            "max": "100+: Diminishing returns"
        },
        "default": 50
    },
    "guidance_scale": {
        "range": "1-20 (typical)",
        "description": "How closely to follow the prompt",
        "ranges": {
            "creative": "1-3: More creative, less prompt-adherent",
            "balanced": "7-8: Balanced (default is 7.5)",
            "literal": "15-20: Very literal, less creative"
        },
        "default": 7.5
    },
    "width": {
        "range": "64-1024 (must be multiple of 64)",
        "description": "Image width in pixels",
        "common_values": [512, 768],
        "note": "Larger = more VRAM needed",
        "default": 512
    },
    "height": {
        "range": "64-1024 (must be multiple of 64)",
        "description": "Image height in pixels",
        "common_values": [512, 768],
        "note": "Larger = more VRAM needed",
        "default": 512
    }
}

# Default parameter sets for different use cases
DEFAULT_PARAMS = {
    "fast": {
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "width": 512,
        "height": 512
    },
    "balanced": {
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "width": 512,
        "height": 512
    },
    "quality": {
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "width": 512,
        "height": 512
    }
}