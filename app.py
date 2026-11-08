"""Web app: upload a photo and the model says whether it shows a cat or a dog.

Run from the project folder:
    python app.py
then open http://127.0.0.1:7860 in your browser.

This is the Python version for running locally. The online version is the browser-only page in
web/ (https://huggingface.co/spaces/bebird/cat-or-dog), which runs the same model exported to ONNX.

The look follows arcade-os-DESIGN.md: a retro desktop with square, bevelled windows,
a limited palette and a pixel font.
"""
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download

from src.predict import Predictor

MODEL_PATH = Path("models/best_model.pt")
# Where the trained model is published on Hugging Face, for machines that don't have it locally
MODEL_REPO = "bebird/cats-vs-dogs-efficientnet-b0"


def find_model():
    """Use the local model after training; otherwise (such as on Hugging Face Spaces) download it."""
    if MODEL_PATH.exists():
        return MODEL_PATH
    return hf_hub_download(repo_id=MODEL_REPO, filename="best_model.pt")


# Load the model once when the app starts, not on every upload
predictor = Predictor(find_model())


def classify(image):
    """Gradio calls this with the uploaded photo and shows what it returns."""
    if image is None:
        return None
    return predictor.predict(image)  # e.g. {"Cat": 0.02, "Dog": 0.98}


# --- Look and feel (colours and rules from arcade-os-DESIGN.md) ---------------------------

PIXEL_FONT = "'Press Start 2P', 'Silkscreen', ui-monospace, Consolas, monospace"
MONO_FONT = "ui-monospace, 'SF Mono', Menlo, Consolas, monospace"

INK, CREAM, CREAM2 = "#1b1230", "#f4ecd0", "#e2d6a8"
LIGHT, SHADE = "#fffbe8", "#b8aa7a"
TEAL, DARK_TEAL = "#2a9d9d", "#0e4f55"
MAGENTA, YELLOW = "#e0338f", "#ffd23f"

# Gradio's theme handles the colours of its own parts (buttons, inputs, panels).
# The same values are set for dark mode, so the retro palette looks identical in both.
_colours = dict(
    body_background_fill=DARK_TEAL,
    body_text_color=INK,
    background_fill_primary=CREAM,
    background_fill_secondary=CREAM2,
    block_background_fill=CREAM,
    block_border_color=INK,
    block_label_background_fill=CREAM2,
    block_label_text_color=INK,
    block_title_text_color=INK,
    border_color_primary=INK,
    input_background_fill=LIGHT,
    button_primary_background_fill=MAGENTA,
    button_primary_background_fill_hover="#c72a7d",
    button_primary_text_color=LIGHT,
    button_primary_border_color=INK,
    button_secondary_background_fill=CREAM2,
    button_secondary_background_fill_hover=CREAM,
    button_secondary_text_color=INK,
    button_secondary_border_color=INK,
    color_accent=MAGENTA,
    color_accent_soft=YELLOW,
    stat_background_fill=MAGENTA,  # the confidence bars
)
theme = gr.themes.Base(
    radius_size=gr.themes.sizes.radius_none,  # the design has no rounded corners
    font=[gr.themes.GoogleFont("Press Start 2P"), "ui-monospace", "monospace"],
    font_mono=["ui-monospace", "Consolas", "monospace"],
)
theme.set(
    **_colours,
    # Only some settings have a separate dark-mode version
    **{f"{name}_dark": value for name, value in _colours.items() if hasattr(theme, f"{name}_dark")},
    block_border_width="2px",
    block_shadow="none",
    button_border_width="2px",
    button_large_radius="0px",
    button_small_radius="0px",
)

# Extra rules the theme can't express: bevels, hard shadows, striped title bars, pixel text.
CSS = f"""
.gradio-container {{
    max-width: 1120px !important;
    margin: 0 auto !important;
    font-family: {MONO_FONT} !important;
    font-size: 13px;
    line-height: 1.45;
}}
/* Dithered teal "desktop wallpaper" behind everything */
body, gradio-app {{
    background-color: {DARK_TEAL} !important;
    background-image: radial-gradient({TEAL} 1px, transparent 1px) !important;
    background-size: 4px 4px !important;
}}

/* A window: 2px ink border, light/shade bevel inside, 4px hard shadow outside */
.ao-win {{
    background: {CREAM} !important;
    border: 2px solid {INK} !important;
    box-shadow: inset 2px 2px 0 {LIGHT}, inset -2px -2px 0 {SHADE}, 4px 4px 0 {INK} !important;
    padding: 0 !important;
    gap: 0 !important;
}}
/* Title bar sits flush with the window edges; everything else gets 14px of padding */
.ao-win > .ao-titlebar-wrap,
.ao-titlebar-wrap .html-container,
.ao-titlebar-wrap .prose {{
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    flex-grow: 0 !important;
}}
.ao-win > *:not(.ao-titlebar-wrap) {{
    margin: 14px 14px 0 !important;
    width: auto !important;
    min-width: 0 !important;
    flex-grow: 0 !important;
}}
.ao-win > *:last-child:not(.ao-titlebar-wrap) {{ margin-bottom: 14px !important; }}

/* Title bar: hard-stop magenta stripes, pixel font, fake window buttons */
.ao-titlebar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 28px;
    padding: 0 6px 0 10px;
    border-bottom: 2px solid {INK};
    background: repeating-linear-gradient(0deg, {MAGENTA} 0 3px, #c72a7d 3px 5px);
    color: {LIGHT};
    font-family: {PIXEL_FONT};
    font-size: 10px;
    letter-spacing: 1px;
}}
.ao-titlebar .ao-buttons span {{
    display: inline-block;
    width: 18px;
    height: 16px;
    margin-left: 4px;
    background: {CREAM};
    border: 2px solid {INK};
    color: {INK};
    font-size: 8px;
    line-height: 12px;
    text-align: center;
}}

.ao-heading {{
    font-family: {PIXEL_FONT};
    font-size: 16px;
    line-height: 1.6;
    margin: 0 0 10px;
    color: {INK};
}}
.ao-text {{ font-family: {MONO_FONT}; color: {INK}; margin: 0; }}
.ao-prompt {{ color: {MAGENTA}; font-weight: bold; }}

/* Buttons get the same bevel and hard shadow, and press down when clicked */
.ao-win button.lg, .ao-win button.md {{
    font-family: {PIXEL_FONT} !important;
    font-size: 10px !important;
    min-height: 30px;
    box-shadow: 3px 3px 0 {INK} !important;
}}
.ao-win button.lg:active, .ao-win button.md:active {{
    transform: translate(3px, 3px);
    box-shadow: none !important;
}}
/* Replace Gradio's blue focus ring; keep a visible dashed one for keyboard users */
.ao-win button:focus {{ outline: none !important; border-color: {INK} !important; }}
.ao-win button:focus-visible {{ outline: 2px dashed {INK} !important; outline-offset: 3px; }}

/* Sunken result field and chunky, square prediction bars */
.ao-result {{ box-shadow: inset 2px 2px 0 {SHADE}, inset -2px -2px 0 {LIGHT} !important; }}
.ao-result .output-class {{ font-family: {PIXEL_FONT} !important; font-size: 22px !important; }}
.ao-result .bar {{
    height: 14px !important;
    border: 2px solid {INK} !important;
    border-radius: 0 !important;
    margin-bottom: 6px !important;
}}
.ao-result .confidence-set {{ margin-bottom: 14px !important; }}

/* Terminal-style readout at the bottom of the result window */
.ao-win > .ao-specs {{ margin-top: auto !important; padding: 0 !important; border: none !important; }}
.ao-specs .html-container, .ao-specs .prose {{ padding: 0 !important; margin: 0 !important; }}
.ao-terminal {{
    background: #0b2b30;
    color: #9ff3d0;
    font-family: {MONO_FONT};
    font-size: 12px;
    line-height: 1.6;
    padding: 10px 12px;
    border: 2px solid {INK};
    box-shadow: inset 2px 2px 0 #05080c;
}}
.ao-terminal .ao-prompt {{ color: {YELLOW}; }}

/* System alert box for the limitations note */
.ao-alert .ao-titlebar {{ background: {YELLOW}; color: {INK}; }}

/* Phones: a 16px gutter instead of Gradio's wider default */
@media (max-width: 760px) {{
    .gradio-container {{ padding: 0 16px !important; }}
    .ao-heading {{ font-size: 13px; }}
}}

footer {{ display: none !important; }}
"""

HEAD = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap">'
)


def title_bar(name):
    """The striped bar across the top of a window, with decorative minimise and close buttons."""
    return gr.HTML(
        f'<div class="ao-titlebar"><span>{name}</span>'
        '<span class="ao-buttons" aria-hidden="true"><span>_</span><span>x</span></span></div>',
        padding=False,
        elem_classes="ao-titlebar-wrap",
    )


# --- Layout --------------------------------------------------------------------------------

with gr.Blocks(title="Cat or Dog?") as demo:
    with gr.Column(elem_classes="ao-win"):
        title_bar("CAT_OR_DOG.EXE")
        gr.HTML(
            '<h1 class="ao-heading">Cat or Dog?</h1>'
            '<p class="ao-text"><span class="ao-prompt">&gt;</span> Upload a photo and an EfficientNet-B0 '
            "model will tell you whether it shows a cat or a dog. It was fine-tuned on 19,793 photos "
            "and scored 98.95% on 2,474 test photos it never saw during training.</p>",
            padding=False,
        )

    with gr.Row(equal_height=True):
        with gr.Column(elem_classes="ao-win"):
            title_bar("INPUT.BMP")
            image = gr.Image(type="pil", label="Photo of a cat or a dog", height=320)
            with gr.Row():
                clear = gr.ClearButton(value="Clear")
                submit = gr.Button("Classify", variant="primary")
        with gr.Column(elem_classes="ao-win"):
            title_bar("RESULT.TXT")
            result = gr.Label(num_top_classes=2, label="Prediction", elem_classes="ao-result")
            gr.HTML(
                '<div class="ao-terminal">'
                '<span class="ao-prompt">&gt;</span> model ......... EfficientNet-B0<br>'
                '<span class="ao-prompt">&gt;</span> input ......... 224 x 224 px<br>'
                '<span class="ao-prompt">&gt;</span> trained on .... 19,793 photos<br>'
                '<span class="ao-prompt">&gt;</span> test accuracy . 98.95%'
                "</div>",
                padding=False,
                elem_classes="ao-specs",
            )

    with gr.Column(elem_classes=["ao-win", "ao-alert"]):
        title_bar("SYSTEM ALERT")
        gr.HTML(
            '<p class="ao-text"><b>Limitations:</b> the model only knows cats and dogs, so any other '
            "photo will still be labelled as one of the two. It is least reliable on dark, blurry or "
            "very small photos, and on photos with both animals in them.</p>",
            padding=False,
        )

    clear.add([image, result])
    # Classify straight away on upload, or when the button is pressed
    image.upload(classify, inputs=image, outputs=result, api_name="classify")
    submit.click(classify, inputs=image, outputs=result, api_name=False)


if __name__ == "__main__":
    demo.launch(theme=theme, css=CSS, head=HEAD)
