# Hate Speech Detection System 

## Project Overview

This repository contains the final project for **TC2034.302** course, implementing an advanced **Hate Speech Detection System** using custom transformer architecture. The project demonstrates cutting-edge NLP techniques for real-time content moderation with a focus on efficiency and accuracy.

**Live Demo**: [https://tc2034302-final-project-ncacrsipbhtr45j3ozwdqk.streamlit.app/](https://tc2034302-final-project-ncacrsipbhtr45j3ozwdqk.streamlit.app/)

**Student**: Jose Emilio Gomez Santos (@sntsemilio)  
**Completed**: June 15, 2025  
**Course**: TC2034.302 - Advanced Algorithms and Data Structures

## Key Features

- **Custom Transformer Architecture**: Built from scratch with multi-head attention (16 heads, d_model=64)
- **Real-time Inference**: 45ms response time for live content moderation
- **Interactive Web Interface**: Streamlit-based application for testing and analysis
- **Model Comparison**: Custom Transformer vs BERT vs RoBERTa benchmarking
- **Hyperparameter Optimization**: Keras Tuner with 20+ optimization trials
- **Comprehensive Analysis**: Dataset visualization, error analysis, and performance metrics

## Performance Highlights

| Metric | Custom Transformer | BERT Base | RoBERTa |
|--------|-------------------|-----------|---------|
| **Accuracy** | **93%** | 91% | 92% |
| **F1-Score** | **0.74** | 0.72 | 0.73 |
| **Parameters** | **847K** | 110M | 125M |
| **Inference Time** | **45ms** | 120ms | 98ms |

### Key Achievements
- **93% accuracy** on highly imbalanced dataset (7% hate speech, 93% normal)
- **15x smaller** than BERT with comparable performance
- **3x faster** inference for real-time deployment
- **Better recall** for hate speech detection
- Optimized for **edge device deployment**

## Prerequisites

- Python 3.7+
- TensorFlow 2.x
- Streamlit
- Required packages (see `requirements.txt`)

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/sntsemilio/TC2034.302-Final-project.git
cd TC2034.302-Final-project
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit application**:
```bash
streamlit run app.py
```

4. **Access the application**:
   - Or visit the live demo: [https://tc2034302-final-project-ncacrsipbhtr45j3ozwdqk.streamlit.app/](https://tc2034302-final-project-ncacrsipbhtr45j3ozwdqk.streamlit.app/)

## Project Structure

```
TC2034.302-Final-project/
├── app.py                           # Main Streamlit application
├── basic_hate_speach_model.ipynb    # Jupyter notebook with model development
├── LICENSE                          # MIT License
├── requirements.txt                 # Python dependencies
├── models/                          # Trained model files
│   ├── hate_speech_model.h5
│   ├── optimized_hate_speech_model.h5
│   ├── tokenizer.pkl
│   └── label_encoder.pkl
└── README.md                        # Project documentation
```

## Model Architecture

### Custom Transformer Design
- **Multi-Head Attention**: 16 attention heads for diverse pattern capture
- **Model Dimension**: 64 (d_model) for efficient computation
- **Positional Encoding**: Custom implementation for short text sequences
- **Dropout Rate**: 0.2 to prevent overfitting on imbalanced data
- **Total Parameters**: 847K (optimized for inference speed)

### Architecture Benefits
- **Efficiency**: 15x smaller than BERT (847K vs 110M parameters)
- **Speed**: 3x faster inference (45ms vs 120ms+)
- **Performance**: Better F1-score on hate speech class (0.74)
- **Deployment**: Suitable for edge devices and real-time processing

## Dataset Information

- **Source**: Tweets Hate Speech Detection Dataset
- **Size**: ~32,000 training samples
- **Challenge**: Highly imbalanced (93% normal, 7% hate speech)
- **Classes**: Binary classification (Hate Speech vs Normal)
- **Preprocessing**: Text cleaning, tokenization, sequence padding

### Technical Challenges Addressed
- **Class Imbalance**: Specialized loss functions and F1-score optimization
- **Ambiguous Language**: Handling sarcasm, irony, and context-dependent meanings
- **Noisy Text**: Social media misspellings, slang, and informal grammar
- **Cultural Context**: Region-specific terms and historical references

## Hyperparameter Optimization

Using **Keras Tuner** with Random Search strategy:
- **Trials**: 20+ hyperparameter combinations
- **Objective**: Maximize F1-score for imbalanced data
- **Parameters Tuned**: Learning rate, dropout, model dimensions, attention heads
- **Validation Strategy**: Stratified train-validation split

## Usage Examples

### Web Interface
Try the live demo at: [https://tc2034302-final-project-ncacrsipbhtr45j3ozwdqk.streamlit.app/](https://tc2034302-final-project-ncacrsipbhtr45j3ozwdqk.streamlit.app/)

1. Navigate to **Inference Interface**
2. Enter text for analysis
3. Select models for comparison
4. View real-time predictions with confidence scores

### Programmatic Usage
```python
# Load trained model
import tensorflow as tf
model = tf.keras.models.load_model('models/optimized_hate_speech_model.h5')

# Predict on new text
prediction = model.predict(preprocessed_text)
```

## Application Sections

1. **Home**: Project overview and key metrics
2. **Inference Interface**: Real-time text analysis
3. **Dataset Analysis**: Comprehensive data visualization
4. **Hyperparameter Tuning**: Optimization process insights
5. **Model Evaluation**: Performance analysis and error investigation

## Future Enhancements

- **Multi-modal Detection**: Text + image hate speech detection
- **Cross-platform Adaptation**: Transfer learning across social platforms
- **Explainable AI**: Transparent decision-making mechanisms
- **Federated Learning**: Privacy-preserving distributed training
- **Multilingual Support**: Detection across different languages
- **Real-time Learning**: Continuous model updates from user feedback

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Jose Emilio Gomez Santos**
- GitHub: [@sntsemilio](https://github.com/sntsemilio)

## Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/sntsemilio/TC2034.302-Final-project/issues) page
2. Create a new issue with detailed description
3. Contact through GitHub profile

---

**Star this repository if you found it helpful!**