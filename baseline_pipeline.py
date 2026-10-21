import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import hstack

class BuildNLPBaseline:
    """
    Production-Ready ML Pipeline for Kaggle NLP Competitions.
    """
    def __init__(self, max_features_word=25000, max_features_char=40000):
        self.word_vec = TfidfVectorizer(ngram_range=(1, 3), max_features=max_features_word, sublinear_tf=True)
        self.char_vec = TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', max_features=max_features_char, sublinear_tf=True)
        self.model_ridge = RidgeClassifier(alpha=1.0)
        self.model_nb = MultinomialNB(alpha=0.1)

    def fit_transform_features(self, train_texts, test_texts):
        print("[+] Extracting Word and Char N-Gram TF-IDF Features...")
        X_train_word = self.word_vec.fit_transform(train_texts)
        X_test_word = self.word_vec.transform(test_texts)
        
        X_train_char = self.char_vec.fit_transform(train_texts)
        X_test_char = self.char_vec.transform(test_texts)
        
        X_train = hstack([X_train_word, X_train_char])
        X_test = hstack([X_test_word, X_test_char])
        return X_train, X_test

    def fit_predict(self, train_df, test_df, text_col='text', label_col='label'):
        X_train, X_test = self.fit_transform_features(train_df[text_col], test_df[text_col])
        y_train = train_df[label_col]
        
        print("[+] Training Ensemble Models...")
        self.model_ridge.fit(X_train, y_train)
        self.model_nb.fit(X_train, y_train)
        
        preds_ridge = self.model_ridge.decision_function(X_test)
        preds_nb = self.model_nb.predict_proba(X_test)[:, 1] if hasattr(self.model_nb, 'predict_proba') else self.model_nb.predict(X_test)
        
        # Blended Ensembling
        final_preds = (preds_ridge * 0.6) + (preds_nb * 0.4)
        print("[+] Prediction Pipeline Complete.")
        return final_preds

if __name__ == "__main__":
    print("Baseline ML Toolkit loaded successfully.")
  
