"""
Utility functions for loading and using the hate speech detection models.
This file would contain the actual model loading and prediction functions.
"""

import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
import pandas as pd
import re
import string
from sklearn.preprocessing import LabelEncoder

class MultiHeadAttention(keras.layers.Layer):
    """Custom Multi-Head Attention layer for the transformer model"""
    
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = keras.layers.Dense(d_model)
        self.wk = keras.layers.Dense(d_model)
        self.wv = keras.layers.Dense(d_model)
        self.dense = keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None):
        q, k, v = inputs, inputs, inputs
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)
        return output

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output

class TransformerBlock(keras.layers.Layer):
    """Transformer block with attention and feed-forward network"""
    
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = self.create_feed_forward_network(d_model, dff)
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

    def create_feed_forward_network(self, d_model, dff):
        return keras.Sequential([
            keras.layers.Dense(dff, activation='relu'),
            keras.layers.Dense(d_model)
        ])

    def call(self, inputs, training=None, mask=None):
        attn_output = self.attention(inputs, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class PositionalEncoding(keras.layers.Layer):
    """Positional encoding for transformer inputs"""
    
    def __init__(self, max_seq_len, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pos_encoding = self.create_positional_encoding(max_seq_len, d_model)

    def create_positional_encoding(self, max_seq_len, d_model):
        pos_encoding = np.zeros((max_seq_len, d_model))

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pos_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i + 1 < d_model:
                    pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        return tf.constant(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:seq_len, :]

def preprocess_text(text):
    """Preprocess text for model input"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

def load_model_components():
    """Load the trained model and preprocessing components"""
    try:
        # In a real implementation, you would load your actual saved models
        # model = keras.models.load_model('optimized_hate_speech_model.h5')
        # with open('optimized_tokenizer.pkl', 'rb') as f:
        #     tokenizer = pickle.load(f)
        # with open('optimized_label_encoder.pkl', 'rb') as f:
        #     label_encoder = pickle.load(f)
        
        # For demo purposes, return None (will use mock predictions)
        return None, None, None
    except FileNotFoundError:
        print("Model files not found. Using mock predictions.")
        return None, None, None

def predict_hate_speech(text, model=None, tokenizer=None, label_encoder=None):
    """Predict hate speech probability for given text"""
    if model is None or tokenizer is None or label_encoder is None:
        # Use mock prediction for demonstration
        return mock_predict(text)
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Tokenize and pad
    sequences = tokenizer.texts_to_sequences([processed_text])
    padded_sequences = keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=100, padding='post', truncating='post'
    )
    
    # Predict
    predictions = model.predict(padded_sequences)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    return predicted_class, confidence, predictions[0]

def mock_predict(text):
    """Mock prediction function for demonstration"""
    processed_text = preprocess_text(text)
    
    # Simple heuristic for demonstration
    hate_keywords = ['hate', 'stupid', 'idiot', 'kill', 'die', 'moron', 'ugly', 'fat']
    
    hate_score = 0.1  # Base probability
    for keyword in hate_keywords:
        if keyword in processed_text.lower():
            hate_score += 0.3
    
    hate_score = min(0.95, hate_score)
    normal_score = 1 - hate_score
    
    predicted_class = 1 if hate_score > 0.5 else 0
    confidence = max(hate_score, normal_score)
    
    return predicted_class, confidence, [normal_score, hate_score]