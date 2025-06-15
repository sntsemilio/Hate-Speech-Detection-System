"""
Data loading and preprocessing utilities for the hate speech detection system.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
import streamlit as st

@st.cache_data
def load_hate_speech_dataset():
    """Load the hate speech detection dataset"""
    try:
        # Load the actual dataset from Hugging Face
        dataset = load_dataset("tweets-hate-speech-detection/tweets_hate_speech_detection")
        
        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        
        return train_df, test_df
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        return create_sample_dataset()

def create_sample_dataset():
    """Create a sample dataset for demonstration purposes"""
    np.random.seed(42)
    
    # Sample normal tweets
    normal_tweets = [
        "Beautiful day for a walk in the park",
        "Just finished an amazing book, highly recommend!",
        "Love spending time with family and friends",
        "Great job on the presentation today",
        "Looking forward to the weekend",
        "This coffee tastes amazing",
        "Had a wonderful time at the concert",
        "Thank you for your help with the project",
        "Excited about the new opportunities ahead",
        "The sunset looks incredible tonight"
    ] * 100  # Repeat to create more samples
    
    # Sample hate speech tweets (censored for safety)
    hate_tweets = [
        "[Content filtered - contains hate speech]",
        "[Content filtered - inappropriate language]",
        "[Content filtered - offensive content]",
        "[Content filtered - discriminatory language]",
        "[Content filtered - threatening language]"
    ] * 20  # Fewer samples to simulate imbalance
    
    # Combine datasets
    all_tweets = normal_tweets + hate_tweets
    labels = [0] * len(normal_tweets) + [1] * len(hate_tweets)
    
    # Shuffle
    combined = list(zip(all_tweets, labels))
    np.random.shuffle(combined)
    tweets, labels = zip(*combined)
    
    df = pd.DataFrame({
        'tweet': tweets,
        'label': labels
    })
    
    # Split into train and test
    split_idx = int(0.8 * len(df))
    train_df = df[:split_idx].copy()
    test_df = df[split_idx:].copy()
    
    return train_df, test_df

def get_dataset_statistics(df):
    """Calculate dataset statistics"""
    stats = {
        'total_samples': len(df),
        'hate_speech_samples': len(df[df['label'] == 1]),
        'normal_samples': len(df[df['label'] == 0]),
        'imbalance_ratio': len(df[df['label'] == 0]) / len(df[df['label'] == 1]) if len(df[df['label'] == 1]) > 0 else 0,
        'average_length': df['tweet'].str.len().mean(),
        'min_length': df['tweet'].str.len().min(),
        'max_length': df['tweet'].str.len().max()
    }
    return stats