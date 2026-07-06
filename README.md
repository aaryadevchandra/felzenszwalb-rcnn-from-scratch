<div id="top"></div>

<br />
<div align="center">
  <h1 align="center">Felzenszwalb R-CNN from Scratch</h1>

  <p align="center">
    A from-scratch R-CNN style object detection pipeline using Felzenszwalb graph-based segmentation for region proposals and a CNN binary classifier for car detection.
    <br />
    <br />
    <a href="https://github.com/krishachemburkar/felzenszwalb-rcnn-from-scratch"><strong>View Repository »</strong></a>
  </p>
</div>

---

## About The Project

This project implements a simple R-CNN style object detection pipeline from scratch.

Instead of using a deep region proposal network, this project uses **Felzenszwalb graph-based segmentation** to generate region proposals. Each proposed region is converted into a bounding box, cropped from the original image, resized, and passed into a CNN classifier to decide whether the crop contains a car.

The goal of this project is to understand the classical object detection pipeline before moving to modern detectors like Faster R-CNN, YOLO, SSD, and DETR.

<p align="right">(<a href="#top">back to top</a>)</p>

---

## Pipeline

```text
Input image
    ↓
Felzenszwalb graph-based segmentation
    ↓
Region proposals / bounding boxes
    ↓
Crop proposed regions
    ↓
Resize crops to 100 × 100
    ↓
CNN binary classifier
    ↓
Car / not-car prediction
    ↓
Final detected boxes
```

<p align="right">(<a href="#top">back to top</a>)</p>

---

## Features

- Felzenszwalb-style graph-based segmentation implemented from scratch
- Region proposal generation using connected components
- Bounding box extraction from segmented regions
- Binary CNN classifier for car vs not-car classification
- PyTorch training pipeline
- Train/test split support
- Loss and accuracy tracking
- Mac MPS support for Apple Silicon acceleration
- COCO negative image collection support
- End-to-end R-CNN style detection flow

<p align="right">(<a href="#top">back to top</a>)</p>

---

## Built With

- Python
- NumPy
- OpenCV
- Matplotlib
- PyTorch
- Torchvision
- pycocotools

<p align="right">(<a href="#top">back to top</a>)</p>

---

## Project Structure

```text
felzenszwalb-rcnn-from-scratch/
│
├── rcnn.ipynb
├── dataset/
│   └── train/
│       ├── car/
│       └── not_car/
├── ncars_png/
├── README.md
└── requirements.txt
```

<p align="right">(<a href="#top">back to top</a>)</p>

---

## Core Ideas

### 1. Region Proposal

The image is treated as a graph:

```text
pixels = nodes
neighboring pixels = edges
edge weight = intensity difference between neighboring pixels
```

The algorithm starts with each pixel as its own component and gradually merges similar neighboring components.

After segmentation, each final component is converted into a bounding box:

```text
x = min column
y = min row
w = max_x - min_x + 1
h = max_y - min_y + 1
```

### 2. Classification

Each proposed bounding box is cropped from the original image and resized to:

```text
100 × 100 × 3
```

The crop is then passed into a CNN classifier trained to distinguish:

```text
car     → 1
not car → 0
```

### 3. Detection

During inference:

```text
1. Generate region proposals
2. Crop every proposal
3. Preprocess each crop
4. Pass crop through CNN
5. Keep boxes with high car probability
```

<p align="right">(<a href="#top">back to top</a>)</p>

---

## Installation

Clone the repository:

```sh
git clone https://github.com/krishachemburkar/felzenszwalb-rcnn-from-scratch.git
cd felzenszwalb-rcnn-from-scratch
```

Install dependencies:

```sh
pip install numpy opencv-python matplotlib torch torchvision pycocotools tqdm requests
```

Or create a `requirements.txt`:

```txt
numpy
opencv-python
matplotlib
torch
torchvision
pycocotools
tqdm
requests
```

Then install:

```sh
pip install -r requirements.txt
```

<p align="right">(<a href="#top">back to top</a>)</p>

---

## Dataset Setup

The classifier expects a binary folder structure:

```text
dataset/
    train/
        car/
        not_car/
```

Positive examples should go inside:

```text
dataset/train/car/
```

Negative examples should go inside:

```text
dataset/train/not_car/
```

Negative examples can be collected from COCO images that do not contain cars.

<p align="right">(<a href="#top">back to top</a>)</p>

---

## Training

The model is trained as a binary classifier using image crops.

Example:

```python
dataset = Dataset()

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)
```

Training tracks:

```text
train loss
test loss
train accuracy
test accuracy
```

<p align="right">(<a href="#top">back to top</a>)</p>

---

## Mac / Apple Silicon Support

If running on a Mac with Apple Silicon, PyTorch MPS can be used:

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

Move the network and tensors to the selected device:

```python
network = network.to(device)
train_img = train_img.float().to(device)
train_label = train_label.float().to(device)
```

<p align="right">(<a href="#top">back to top</a>)</p>

---

## Example Detection Flow

```python
img_color = cv2.imread("image.jpg")
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

bboxes = region_proposal(img_gray)

detections = []

for x, y, w, h in bboxes:
    patch = img_color[y:y+h, x:x+w, :]

    patch_tensor = preprocess_patch(patch).to(device)

    prob = network(patch_tensor).view(-1).item()

    if prob > 0.5:
        detections.append((x, y, w, h, prob))
```

<p align="right">(<a href="#top">back to top</a>)</p>

---

## Results

The detector generates candidate boxes using Felzenszwalb segmentation and classifies each crop as car or not-car.

Example output format:

```text
(x, y, width, height, probability)
```

<p align="right">(<a href="#top">back to top</a>)</p>

---

## Limitations

This is a learning-focused implementation, not a production-grade detector.

Current limitations:

- Region proposals are based on segmentation, so boxes may be noisy
- A single segment may not cover the full object
- Many small regions need filtering
- No bounding box regression yet
- No non-max suppression yet
- Performance depends heavily on region proposal quality
- CNN classifier is trained only for binary car detection

<p align="right">(<a href="#top">back to top</a>)</p>

---

## Future Improvements

- Add Non-Max Suppression
- Add bounding box regression
- Improve region merging after Felzenszwalb segmentation
- Add IoU-based positive and negative proposal labeling
- Train on VOC or COCO car annotations
- Compare with Selective Search
- Add evaluation metrics such as precision, recall, and mAP
- Convert notebook code into modular Python files

<p align="right">(<a href="#top">back to top</a>)</p>

---

## Learning Motivation

This project was built to understand how early object detection pipelines worked before modern end-to-end detectors.

It focuses on the core R-CNN idea:

```text
region proposal + crop classifier = object detector
```

By implementing segmentation, bounding boxes, preprocessing, and classification manually, this project helps build intuition for how object detection systems work under the hood.

<p align="right">(<a href="#top">back to top</a>)</p>

---

## License

This project is open source and available under the MIT License.

<p align="right">(<a href="#top">back to top</a>)</p>

---
