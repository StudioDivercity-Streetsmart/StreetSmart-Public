# Streetsmart
A digital street rating and redesign platform empowering local governments and communities
to make cities more liveable & climate resilient Core Application and Functionality

## MISSION
Enable a world where streets are designed for people, not cars


---


The Streetsmart project integrates multiple AI modules to enhance urban planning through advanced data processing and machine learning. The key functionalities include:

- **Urban Planning Expert:**  
  Gives insights and recommendations based on an image of a street or public space that could be improved.

- **Semantic Segmentation:**  
  Overlays segmentation masks on images, applies a color palette to different classes, and saves the annotated output. Interfaces with the Hugging Face API to generate semantic segmentation maps.

- **Object Detection:**  
  This module processes images to detect objects, draws bounding boxes with labels, and outputs object data in JSON format. Uses a TensorFlow Hub module to perform object detection on images. 

Each Python file corresponds to a specific module and can be executed independently using a simple command.

---

## How to Run the Python Files

Get the required Python packages
``pipenv install``
``pipenv shell``

Add your API keys to a .env file:
```
OPENAI_API_KEY=your-api-key-here
HUGGINGFACE_API_TOKEN=your-api-key-here
```

Run each module using the following commands:

Urban Planning Expert
```bash
python backend/src/ai/chatgpt_client.py
```
Semantic Segmentation
```bash
python backend/src/ai/semantic-segmentation.py
```
Object Detection
```bash
python backend/src/ai/object-detection-label.py
```



