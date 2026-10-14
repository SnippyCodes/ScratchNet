import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import gradio as gr
from PIL import Image
from model import predict_flower, CATEGORIES, FLOWER_DETAILS


def classify_flower(image):
    if image is None:
        return "Please upload an image.", {}, ""

    res = predict_flower(image)
    top_name = res["display_name"]
    confidence = res["confidence_pct"]
    info = res["flower_info"]
    icon = info.get("icon", "🌸")
    sci_name = info.get("scientific_name", "")

    # Top prediction badge text
    title_text = f"## {icon} {top_name} ({confidence})\n*{sci_name}*"

    # Dictionary of label -> probability for Gradio Label component
    prob_dict = {item["display_name"]: item["probability"] for item in res["ranked_list"]}

    # Botanical details
    details_md = f"""
### 🌿 Botanical Profile:
- **Family:** {info.get('family', 'N/A')}
- **Blooms:** {info.get('season', 'N/A')}
- **Description:** {info.get('description', '')}

*Inference Hardware:* `{res['device'].upper()}`
    """

    return title_text, prob_dict, details_md


# Gather sample examples
examples = []
sample_dir = Path("static/samples")
if sample_dir.is_dir():
    for f in sorted(sample_dir.glob("*.jpg")):
        examples.append([str(f)])

with gr.Blocks(title="ScratchNet — 7-Class Flower Classifier") as demo:
    gr.Markdown("""
    # 🪷 ScratchNet AI — 7-Class Flower Classifier
    Upload or capture a photo of a flower to classify between **Daisy, Dandelion, Rose, Sunflower, Tulip, Bellflower, and Lotus**.
    Powered by a custom 3-block 2D Convolutional Neural Network (PyTorch).
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Flower Image")
            predict_btn = gr.Button("🌸 Classify Flower", variant="primary")
            if examples:
                gr.Examples(examples=examples, inputs=input_image, label="Verified Flower Samples")

        with gr.Column():
            output_title = gr.Markdown(label="Predicted Flower")
            output_labels = gr.Label(num_top_classes=7, label="Class Probabilities")
            output_details = gr.Markdown(label="Botanical Insights")

    predict_btn.click(
        fn=classify_flower,
        inputs=input_image,
        outputs=[output_title, output_labels, output_details],
    )
    input_image.change(
        fn=classify_flower,
        inputs=input_image,
        outputs=[output_title, output_labels, output_details],
    )

if __name__ == "__main__":
    print("Launching Gradio interface...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
