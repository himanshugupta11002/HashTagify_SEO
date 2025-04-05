import os
import re
import torch
import requests
from flask import Flask, render_template, request, send_from_directory
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from io import BytesIO
import random

app = Flask(__name__)

#Load all models

detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


classification_model = models.resnet50(pretrained=True)
classification_model.eval()

captioning_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
captioning_feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
captioning_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
captioning_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
captioning_model.to(captioning_device)

# Define ImageNet transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Map ImageNet class indices to class labels
imagenet_labels = requests.get(
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json").json()

# Set the uploads folder
UPLOADS_FOLDER = 'uploads'
app.config['UPLOADS_FOLDER'] = UPLOADS_FOLDER
os.makedirs(os.path.join(app.instance_path, UPLOADS_FOLDER), exist_ok=True)

stop_words = set(stopwords.words('english'))


# Function to generate SEO-friendly synonyms
def get_seo_friendly_synonyms(word):
    synonyms = set()

    # Use WordNet to get synonyms for the word
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower():
                synonyms.add(synonym)

    # Add SEO-friendly words list as additional synonyms
    seo_friendly_synonyms = list(synonyms)[:3]  # Limit to top 3 synonyms for variety
    return seo_friendly_synonyms


# Function to convert caption into SEO-friendly words and hashtags
def convert_caption_to_seo_friendly_words_and_hashtags(caption):
    words = re.findall(r'\w+', caption.lower())  # Tokenize the caption
    seo_friendly_words = []

    # Replace each word with its SEO-friendly synonyms
    for word in words:
        if word not in stop_words and len(word) > 2:  # Filter out stopwords and short words
            seo_friendly_synonyms = get_seo_friendly_synonyms(word)
            if seo_friendly_synonyms:
                seo_friendly_words.extend(seo_friendly_synonyms)
            else:
                seo_friendly_words.append(word)

    # Deduplicate and get the top SEO-friendly words
    seo_friendly_words = list(set(seo_friendly_words))[:10]

    # Generate hashtags
    hashtags = ['#' + word for word in seo_friendly_words]

    return ' '.join(seo_friendly_words), ' '.join(hashtags)


# Hugging Face image captioning prediction
def predict_step(image_path):
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    images = [i_image]

    pixel_values = captioning_feature_extractor(images=images, return_tensors="pt",
                                                return_attention_mask=True).pixel_values
    pixel_values = pixel_values.to(captioning_device)

    output_ids = captioning_model.generate(pixel_values)

    preds = captioning_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    seo_friendly_caption, hashtags = convert_caption_to_seo_friendly_words_and_hashtags(preds[0])

    return seo_friendly_caption, hashtags


# Object detection using YOLO
def detect_objects(image_path, confidence_threshold=0.1):
    img = Image.open(image_path)
    results = detection_model(img, size=640)  # Adjust the size parameter as needed

    # Filter results based on confidence threshold
    filtered_results = results.xyxy[0][results.xyxy[0][:, 4] > confidence_threshold]
    return filtered_results.cpu().numpy()


# Image classification using ResNet50
def classify_image(image_path):
    img = Image.open(image_path)
    img = preprocess(img)
    img = img.unsqueeze(0)

    # Perform classification
    with torch.no_grad():
        output = classification_model(img)

    # Get the predicted class index
    _, predicted_idx = torch.max(output, 1)

    # Get the predicted class label
    predicted_label = imagenet_labels[predicted_idx.item()]

    return predicted_label


# Process and save image with detection, classification, and captioning
# def process_and_save_image(original_image_path, detection_results, classification_results, caption_results, hashtags):
#     img = Image.open(original_image_path)
#
#     # Create a drawing object
#     draw = ImageDraw.Draw(img)
#
#     # Draw bounding boxes and add captions based on detection, classification, and captioning results
#     for result in detection_results:
#         box = result[:4]
#
#         # Convert box coordinates to integers
#         box = [int(coord) for coord in box]
#
#         # Draw bounding box
#         draw.rectangle(box, outline="red", width=2)
#
#     # Add classification, captioning, and hashtags text
#     draw.text((10, 10), f"Classification: {classification_results}", fill="blue")
#     draw.text((10, 30), f"Caption: {caption_results}", fill="green")
#     draw.text((10, 50), f"Hashtags: {hashtags}", fill="purple")
#
#     # Save the processed image with detection results, classification, and caption
#     output_image_path = f"processed_{os.path.basename(original_image_path)}"
#     img.save(os.path.join(app.config['UPLOADS_FOLDER'], output_image_path))
#
#     return output_image_path
def process_and_save_image(original_image_path, detection_results, classification_results, caption_results, hashtags):
    img = Image.open(original_image_path)
    draw = ImageDraw.Draw(img)

    # Define font (you might need to specify a font file if the default isn't suitable)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    # Define text content and colors
    text_blocks = [
        (f"Classification: {classification_results}", "blue", (255, 255, 255)),
        (f"Caption: {caption_results}", "green", (255, 255, 255)),
        (f"Hashtags: {hashtags}", "purple", (255, 255, 255))
    ]

    # Initial position
    x, y = 10, 10
    padding = 5
    box_margin = 5

    for text, box_color, text_color in text_blocks:
        # Calculate text size using textbbox (new method)
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
        text_width = right - left
        text_height = bottom - top

        # Draw background rectangle
        draw.rectangle(
            [(x, y), (x + text_width + 2 * padding, y + text_height + 2 * padding)],
            fill=box_color
        )

        # Draw text
        draw.text(
            (x + padding, y + padding),
            text,
            fill=text_color,
            font=font
        )

        # Update y position for next box with margin
        y += text_height + 2 * padding + box_margin

    # Draw bounding boxes for detected objects
    for result in detection_results:
        box = result[:4]
        box = [int(coord) for coord in box]
        draw.rectangle(box, outline="red", width=2)

    # Save the processed image
    output_image_path = f"processed_{os.path.basename(original_image_path)}"
    img.save(os.path.join(app.config['UPLOADS_FOLDER'], output_image_path))

    return output_image_path

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOADS_FOLDER'], filename)


@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        image_files = request.files.getlist('images')
        processed_images = []
        for image_file in image_files:
            # Save the image
            image_path = os.path.join(app.config['UPLOADS_FOLDER'], image_file.filename)
            image_file.save(image_path)
            detection_results = detect_objects(image_path)

            classification_result = classify_image(image_path)

            seo_friendly_caption, hashtags = predict_step(image_path)

            output_image_path = process_and_save_image(image_path, detection_results, classification_result,
                                                       seo_friendly_caption, hashtags)

            processed_images.append(output_image_path)

        return render_template('result.html', processed_images=processed_images)


if __name__ == '__main__':
    app.run(debug=True)
