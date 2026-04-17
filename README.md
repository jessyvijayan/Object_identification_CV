# Object_identification_CV

Real-Time Face Mask Detection: XML Annotation & CNN Pipeline
This project implements a complete Computer Vision pipeline, from custom data extraction and annotation parsing to real-time object classification. The system detects three states: Wearing a Mask, No Mask, and Incorrectly Worn Mask, utilizing a custom Convolutional Neural Network (CNN).

🚀 Key Features
XML Annotation Engineering: Developed a specialized parser using BeautifulSoup to extract bounding boxes and labels from PASCAL VOC XML annotations, converting unstructured spatial data into structured NumPy arrays.
Coordinate Extraction: Automated the retrieval of xmin, ymin, xmax, and ymax attributes for 850+ high-resolution images, facilitating precise region-of-interest (ROI) cropping.
Advanced Data Handling: Implemented a custom data generator that synchronizes image files with their corresponding XML labels, handling complex multi-object scenarios (up to 16+ persons per image).
Multi-Class Detection: Engineered the model to distinguish between three safety states, specifically targeting the "incorrectly worn" edge case to improve real-world safety compliance.

🛠️ Tech Stack
Computer Vision: OpenCV (cv2)
Data Parsing: BeautifulSoup (XML parsing), OS, NumPy
Deep Learning: TensorFlow / Keras (CNN, VGG19 Preprocessing)
Image Augmentation: ImageDataGenerator (for robust model generalization)

📸 Technical Workflow
Annotation Parsing: Utilized BeautifulSoup with the xml parser to iterate through nested <object> tags, mapping string labels (e.g., with_mask) to categorical integers.
Spatial Mapping: Extracted 4-point coordinate arrays to define object boundaries, enabling the model to learn localized facial features.
Preprocessing: Integrated vgg19.preprocess_input to standardize image tensors, ensuring compatibility with high-performance feature extraction layers.
CNN Architecture: Built a Sequential model featuring Conv2D for feature mapping and MaxPooling2D for spatial variance reduction.

📈 Model Capabilities
High-Density Detection: Capable of identifying and labeling multiple individuals in crowded frames (as demonstrated in the data extraction logs).
Spatial Awareness: The integration of bounding box coordinates ensures the model focuses strictly on facial regions, reducing noise from background elements.

💻 How to Run
Clone the repository:
git clone [git@github.com:jessyvijayan/Object_identification_CV.git]

Install Dependencies:
pip install tensorflow opencv-python beautifulsoup4 numpy matplotlib lxml

Execute:
[python mask_detection_main.py](https://github.com/jessyvijayan/Object_identification_CV/blob/main/Object_identification_computer_vision.py)


