import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import base64
from io import BytesIO

# Load pretrained ResNet50 model
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

# Feature extraction
def extract_features(img):
    img = img.convert("RGB")  # ensure 3 channels
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

# Cosine similarity
def cosine_similarity(vec1, vec2):
    epsilon = 1e-10
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + epsilon)

# Convert PIL image to base64 for HTML display
def pil_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Compute similarity with vibrant HTML
def compute_similarity_html(img1, img2):
    f1 = extract_features(img1)
    f2 = extract_features(img2)
    score = cosine_similarity(f1, f2) * 100
    score = round(score, 2)

    if score > 70:
        bar_color = "linear-gradient(90deg, #00C853, #B2FF59)"
        emoji = "✅"
        show_confetti = True
    elif score > 40:
        bar_color = "linear-gradient(90deg, #FFD600, #FFFF8D)"
        emoji = "⚠️"
        show_confetti = False
    else:
        bar_color = "linear-gradient(90deg, #D50000, #FF8A80)"
        emoji = "❌"
        show_confetti = False

    img1_url = pil_to_base64(img1)
    img2_url = pil_to_base64(img2)

    confetti_script = ""
    if show_confetti:
        confetti_script = """
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
        <script>
        confetti({ particleCount: 150, spread: 70, origin: { y: 0.6 } });
        </script>
        """

    html = f"""
    <div style="
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
        max-width:700px; margin:auto; text-align:center; 
        padding:30px; border-radius:20px; 
        background: #F5F5DC; /* beige background */
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        color: #333;
        display: flex;
        flex-direction: column;
        align-items: center;
    ">
        <h1 style="color:#6B4C3B; font-size:2.2em; margin-bottom:20px;">
            Snap, Upload, Compare-- Upload Your Image & See How Similar It Is!
        </h1>
        <div style="display:flex; justify-content:center; gap:25px; margin-bottom:25px;">
            <img src="{img1_url}" style="width:45%; border-radius:15px; box-shadow:0 6px 15px rgba(0,0,0,0.2); transition: transform 0.3s;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'"/>
            <img src="{img2_url}" style="width:45%; border-radius:15px; box-shadow:0 6px 15px rgba(0,0,0,0.2); transition: transform 0.3s;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'"/>
        </div>
        <div style="background:#eee; border-radius:25px; overflow:hidden; height:35px; margin-bottom:20px; width:100%;">
            <div style="
                width:0%; height:100%; text-align:center; 
                line-height:35px; font-weight:bold; color:white; 
                background:{bar_color}; animation: fillBar 1.5s forwards;
            ">
                {score}% {emoji}
            </div>
        </div>
        <p style="font-size:16px;">
            <strong>Explanation:</strong> Similarity is calculated using deep learning features from ResNet50 and cosine similarity.
        </p>
    </div>

    {confetti_script}

    <style>
        @keyframes fillBar {{
            from {{ width: 0%; }}
            to {{ width: {score}%; }}
        }}
    </style>
    """
    return html

# --- Gradio Blocks with full-screen beige background ---
with gr.Blocks(css="""
    html, body, .gradio-container {
        height: 100%;
        margin: 0;
        padding: 0;
        background: #F5F5DC; /* full-page beige background */
        display: flex;
        justify-content: center;
        align-items: flex-start;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .gradio-container {
        width: 100%;
    }
    .gr-button { 
        background: linear-gradient(45deg, #6B4C3B, #8D6E63); 
        color: white; 
        font-weight:bold;
        font-size:16px;
        border-radius:12px;
        padding:12px 25px;
        box-shadow:0 5px 15px rgba(0,0,0,0.2);
        margin-top:10px;
    }
    .gr-button:hover {
        opacity:0.9;
        transform: scale(1.05);
        transition:0.3s;
    }
""") as demo:

    gr.Markdown("<h1 style='text-align:center; color:#6B4C3B; font-size:2.5em; margin-bottom:30px;'>Snap, Upload, Compare-- Upload Your Image & See How Similar It Is!</h1>")
    
    with gr.Row():
        img1_input = gr.Image(type="pil", label="Upload Image 1")
        img2_input = gr.Image(type="pil", label="Upload Image 2")
    
    output_html = gr.HTML()
    
    compute_btn = gr.Button("Compute Similarity")
    compute_btn.click(fn=compute_similarity_html, inputs=[img1_input, img2_input], outputs=output_html)

demo.launch()
