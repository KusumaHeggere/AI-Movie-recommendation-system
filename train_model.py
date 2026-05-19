import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time

def train_sentiment_model():
    print("="*50)
    print("  CineMatch ML Training Pipeline")
    print("="*50)
    
    start_time = time.time()
    
    print("\n[1/5] Downloading IMDb Movie Reviews Dataset (Hugging Face)...")
    # Load dataset
    dataset = load_dataset("imdb")
    
    # We will combine train and test, then sample 15,000 reviews for speed & efficiency
    # 1 is Positive, 0 is Negative
    df_train = pd.DataFrame(dataset['train'])
    df_test = pd.DataFrame(dataset['test'])
    df_all = pd.concat([df_train, df_test])
    
    # Shuffle and sample
    df_sample = df_all.sample(n=15000, random_state=42)
    
    X = df_sample['text']
    y = df_sample['label']
    
    print(f"      -> Successfully loaded {len(X)} reviews!")

    print("\n[2/5] Vectorizing Text (Converting words to math)...")
    # Initialize TF-IDF Vectorizer to limit to top 10,000 most important words
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
    X_vectorized = vectorizer.fit_transform(X)
    print(f"      -> Created a matrix of shape: {X_vectorized.shape}")

    print("\n[3/5] Training Logistic Regression AI Model...")
    # Train the model
    model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    model.fit(X_vectorized, y)
    print("      -> Training complete!")
    
    print("\n[4/5] Evaluating Model Performance...")
    # Quick self-evaluation
    predictions = model.predict(X_vectorized)
    acc = accuracy_score(y, predictions)
    print(f"      -> Internal Accuracy: {acc * 100:.2f}%")

    print("\n[5/5] Saving 'Brain' Weights to disk...")
    # Save the vectorizer and the model
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    joblib.dump(model, 'sentiment_model.pkl')
    print("      -> Saved 'tfidf_vectorizer.pkl'")
    print("      -> Saved 'sentiment_model.pkl'")

    elapsed = time.time() - start_time
    print(f"\nPipeline Finished in {elapsed:.1f} seconds!")
    print("   The ML Model is now ready to be plugged into Flask!")

if __name__ == "__main__":
    train_sentiment_model()
