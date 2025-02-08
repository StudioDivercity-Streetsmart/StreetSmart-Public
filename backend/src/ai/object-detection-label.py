# For running inference on the TF-Hub module.
import tensorflow as tf
import json
import tensorflow_hub as hub

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

def display_image(image):
  # This function is kept for compatibility but does nothing
  pass


def download_and_resize_image(path, new_width=256, new_height=256, display=False):
  _, filename = tempfile.mkstemp(suffix=".jpg")
  
  # Check if the path is a URL or local file
  if path.startswith('http://') or path.startswith('https://'):
    response = urlopen(path)
    image_data = response.read()
    image_data = BytesIO(image_data)
    pil_image = Image.open(image_data)
    print("Image downloaded from URL")
  else:
    # Open local file directly
    pil_image = Image.open(path)
    print("Image opened from local file")
  
  pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.LANCZOS)
  pil_image_rgb = pil_image.convert("RGB")
  pil_image_rgb.save(filename, format="JPEG", quality=90)
  print("Image processed and saved to %s." % filename)
  
  if display:
    display_image(pil_image)
  return filename


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getbbox(ds)[3] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    bbox = font.getbbox(display_str)
    text_width, text_height = bbox[2], bbox[3]
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())
  image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
  font = ImageFont.load_default()

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                     int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      draw_bounding_box_on_image(
          image_pil,
          ymin,
          xmin,
          ymax,
          xmax,
          color,
          font,
          display_str_list=[display_str])
  
  return np.array(image_pil)

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img

def run_detector(detector, path, output_path):
  img = load_img(path)

  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  start_time = time.time()
  result = detector(converted_img)
  end_time = time.time()

  result = {key: value.numpy() for key, value in result.items()}

  print("Found %d objects." % len(result["detection_scores"]))
  print("Inference time: ", end_time - start_time)

  # Draw boxes on image without displaying it
  image_with_boxes = draw_boxes(
      img.numpy(), result["detection_boxes"],
      result["detection_class_entities"], result["detection_scores"])

  # Save the image with boxes
  image_pil = Image.fromarray(np.uint8(image_with_boxes))
  image_pil.save(output_path)
  print(f"Saved annotated image to: {output_path}")

  # Create a list to store object data
  objects = []
  for i in range(min(result["detection_boxes"].shape[0], 10)):
    if result["detection_scores"][i] >= 0.1:
      ymin, xmin, ymax, xmax = tuple(result["detection_boxes"][i])
      width = xmax - xmin
      height = ymax - ymin
      x = xmin
      y = ymin
      label = result["detection_class_entities"][i].decode("ascii")
      objects.append({
          "width": float(width),
          "height": float(height),
          "x": float(x),
          "y": float(y),
          "label": label
      })

  # Convert object data to JSON string
  json_string = json.dumps(objects, indent=2)
  return json_string



if __name__ == "__main__":

    # For measuring the inference time.
    import time
    # Print Tensorflow version
    print(tf.__version__)
    # Check available GPU devices.
    print("The following GPU devices are available: %s" % tf.test.gpu_device_name())
    # Use local image file (ensure the path is correct on your system)
    input_image = 'backend/src/ai/files/14_-_Alzaibar_entre_Buenos_Aires_y_Sarandi_hacia_el_Sur.jpg'
    # Set display to False to prevent any image from being shown
    processed_image_path = download_and_resize_image(input_image, 1280, 856, display=False)

    # faster_rcnn
    ## "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    # mobile net
    ## https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1
    module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"

    detector = hub.load(module_handle).signatures['default']
    # Set output path for the annotated image
    output_image_path = 'out.png'

    # Run detector and save output image and JSON (no image display)
    json_string = run_detector(detector, processed_image_path, output_image_path)
    print(json_string)

