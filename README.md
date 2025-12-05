# Image Caption Generator Using Deep Learning

A deep learning-based application that automatically generates descriptive captions for images using CNN-LSTM architecture. The model leverages pre-trained convolutional neural networks (ResNet50/MobileNetV2) for feature extraction and LSTM networks for sequence generation.

## 🎯 Objectives

- Recognize and understand the context of images
- Generate accurate and meaningful English captions automatically
- Provide an intuitive user interface for easy interaction

## 🌟 Features

- **Multiple Model Support**: Utilizes ResNet50 and MobileNetV2 for robust image feature extraction
- **LSTM-based Caption Generation**: Employs Long Short-Term Memory networks for sequential text generation
- **Interactive Web Interface**: Built with Streamlit for easy image upload and caption generation
- **Tkinter GUI**: Alternative desktop application interface
- **Pre-trained Models**: Comes with trained models ready for inference

## 🏗️ Architecture

![System Diagram](images/systemdiagram.PNG)

### Model Components

1. **CNN (Convolutional Neural Network)**: Extracts visual features from images
   - ResNet50: Extracts 2048-dimensional feature vectors
   - MobileNetV2: Lightweight alternative for faster inference
   
2. **LSTM (Long Short-Term Memory)**: Generates captions from image features
   - Processes sequential data
   - Learns long-term dependencies in language

3. **Tokenizer**: Converts words to numerical indices and vice versa

### How It Works

```
Image → CNN (ResNet50) → Feature Vector (2048) → LSTM → Caption Generation
                                    ↓
                            Word Embeddings + Sequence Processing
                                    ↓
                            Probability Distribution (Softmax)
                                    ↓
                            Greedy Search → Final Caption
```

## 📊 Dataset

**Flickr30K Dataset**
- 30,000 images
- 5 captions per image (150,000 total captions)
- Alternative datasets: Flickr8K, MSCOCO

### Caption Distribution
![Word Distribution](images/words.JPG)

## 🔄 Data Processing Pipeline

### 1. Data Collection
The model uses the Flickr30K dataset with comprehensive image-caption pairs stored in:
- Images: `data/Images/`
- Captions: `data/textFiles/`

### 2. Data Cleaning
```python
def clean(data):
    # Convert all characters to lower case
    data = data.lower()
    
    # Convert all non-alphabet characters to spaces
    data = re.sub("[^a-z]+", " ", data)
    
    return data

# Example:
clean("A man in green holds a guitar while the other man observes his shirt.")
# Output: "a man in green holds a guitar while the other man observes his shirt"
```

**Cleaning Steps:**
- Lowercase all characters
- Remove non-alphabetic characters (#, %, $, &, @, etc.)
- Save cleaned captions to `tokens_clean.txt`

### 3. Caption Tokenization
Special tokens added to each caption:
- `startseq`: Marks the beginning of a caption
- `endseq`: Marks the end of a caption

Example: `startseq a man in green holds a guitar endseq`

### 4. Image Preprocessing
- Images resized to 224×224 pixels
- Converted to array format
- Normalized using model-specific preprocessing
- Feature extraction produces 2048-dimensional vectors (ResNet50)

### 5. Text Preprocessing
Two key dictionaries created:
- `word_to_index`: Maps words to numerical indices
- `index_to_word`: Maps indices back to words

Usage:
- `word_to_index['guitar']` → returns index of the word 'guitar'
- `index_to_word[42]` → returns the word at index 42

### 6. Generator Function
- Efficiently loads and processes data in batches
- Feeds image features and captions to the LSTM model during training

### 7. Caption Generation (Inference)
- Model outputs probability distribution across vocabulary (softmax)
- Greedy Search Algorithm selects words with maximum probability
- Process repeats until `endseq` token is generated or max length reached

## 📈 Model Training

### Training Configuration
- **Epochs**: 14-15
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy
- **Evaluation Metric**: BLEU Score
- **Max Caption Length**: 80 tokens
- **Training Time**: ~21 hours on Intel i7 8750H

### Model Checkpoints
Trained models are saved in `model_checkpoints/`:
- `model_13.h5`
- `model_14.h5`
- `mymodel.h5` (main deployment model)

## 🖼️ Results

Generated captions on test images:

![Caption Example 1](images/caption3.JPG)

![Caption Example 2](images/caption4.JPG)

### User Interface
Desktop application created using Tkinter:

![UI Screenshot](images/ui.JPG)

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Anaconda (recommended)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd "Image CAPTIONING"
```

2. **Create virtual environment** (optional but recommended)
```bash
conda create -n image-caption python=3.7
conda activate image-caption
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Directory Structure
```
Image CAPTIONING/
├── app.py                          # Streamlit app (MobileNetV2)
├── appCNN.py                       # Streamlit app (ResNet50)
├── ui.py                           # Tkinter desktop GUI
├── mymodel.h5                      # Trained model
├── tokenizer.pkl                   # Tokenizer for app.py
├── requirements.txt                # Python dependencies
├── data/
│   ├── textFiles/
│   │   ├── 30k_captions.txt       # Original captions
│   │   ├── tokens_clean.txt       # Cleaned captions
│   │   ├── flickr30k_train.txt    # Training data
│   │   ├── flickr30k_val.txt      # Validation data
│   │   ├── flickr30k_test.txt     # Test data
│   │   ├── word_to_idx.pkl        # Word to index mapping
│   │   └── idx_to_word.pkl        # Index to word mapping
│   └── Images/                     # Dataset images
├── model_checkpoints/              # Saved models during training
├── images/                         # Screenshots and diagrams
└── notebooks/
    ├── image-captioner.ipynb       # Caption generation notebook
    ├── model_build.ipynb           # Model training notebook
    └── text_data_processing.ipynb  # Data preprocessing notebook
```

## 💻 Usage

### Option 1: Streamlit Web App (MobileNetV2)

```bash
streamlit run app.py
```

1. Open browser at `http://localhost:8501`
2. Upload an image (JPG, JPEG, PNG)
3. View generated caption

### Option 2: Streamlit Web App (ResNet50)

```bash
streamlit run appCNN.py
```

1. Open browser at `http://localhost:8501`
2. Upload an image
3. Click "Generate Caption"
4. View results

### Option 3: Tkinter Desktop GUI

```bash
python ui.py
```

1. Click "Select Image" to choose an image
2. Click "Generate Caption" to generate description
3. View caption in the interface

## 📓 Training Your Own Model

### Step 1: Prepare Dataset
```python
# Run text preprocessing notebook
jupyter notebook text_data_processing.ipynb
```
- Creates word-to-index and index-to-word dictionaries
- Cleans and tokenizes captions
- Splits data into train/val/test sets

### Step 2: Build and Train Model
```python
# Run model building notebook
jupyter notebook model_build.ipynb
```
- Defines CNN-LSTM architecture
- Trains the model on Flickr30K dataset
- Evaluates using BLEU score
- Saves model checkpoints

### Step 3: Test the Model
```python
# Run caption generation notebook
jupyter notebook image-captioner.ipynb
```
- Tests model on custom images
- Visualizes results
- Generates captions for test set

## 📦 Key Dependencies

```
tensorflow==2.15.0
keras==2.15.0
numpy==1.23.5
nltk==3.8.1
Pillow==9.5.0
streamlit
matplotlib==3.7.2
pandas==2.0.3
h5py==3.9.0
```

See `requirements.txt` for complete list.

## 🛠️ Technologies Used

- **Deep Learning Framework**: TensorFlow/Keras
- **CNN Architecture**: ResNet50, MobileNetV2
- **RNN Architecture**: LSTM
- **Web Framework**: Streamlit
- **GUI Framework**: Tkinter
- **Image Processing**: PIL, OpenCV
- **NLP**: NLTK
- **Data Processing**: NumPy, Pandas

## 🔍 Model Evaluation

### BLEU Score
- Evaluates quality of generated captions
- Compares generated captions with reference captions
- Higher score indicates better performance

### Caption Quality Examples

**Good Predictions:**
- Input: Image of a dog playing
- Output: "a dog is playing with a ball in the grass"

**Model Strengths:**
- Identifies common objects accurately
- Understands basic scene composition
- Generates grammatically correct sentences

## ⚙️ Configuration

### Model Parameters
- **Image Size**: 224×224
- **Feature Vector**: 2048 dimensions (ResNet50)
- **Vocabulary Size**: ~8,000 words
- **Max Caption Length**: 80 words
- **Embedding Dimension**: 256
- **LSTM Units**: 256

## 🐛 Troubleshooting

### Common Issues

**1. Model file not found**
```bash
# Ensure model files are in the correct location
ls model_checkpoints/
ls mymodel.h5
```

**2. Missing tokenizer**
```bash
# Check if tokenizer.pkl exists for app.py
ls tokenizer.pkl

# Or word_to_idx.pkl and idx_to_word.pkl for appCNN.py
ls data/textFiles/*.pkl
```

**3. Memory errors during training**
- Reduce batch size
- Use a GPU for training
- Use MobileNetV2 instead of ResNet50

**4. Import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Future Improvements

- [ ] Implement attention mechanism for better caption quality
- [ ] Add beam search for improved caption generation
- [ ] Support for multiple languages
- [ ] Real-time video captioning
- [ ] Fine-tune on domain-specific datasets
- [ ] Deploy as web service with REST API
- [ ] Add more evaluation metrics (METEOR, CIDEr)
- [ ] Implement transformer-based models (Vision Transformer + GPT)

## 📄 License

This project is available for educational and research purposes.

## 👏 Acknowledgments

- Flickr30K dataset creators
- TensorFlow and Keras teams
- Pre-trained model contributors (ResNet50, MobileNetV2)
- Streamlit for the excellent web framework

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: Ensure you have sufficient computational resources (GPU recommended) for training the model from scratch. Pre-trained models are provided for immediate inference.
#
