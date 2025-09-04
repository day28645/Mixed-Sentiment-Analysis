# Thai Sentiment Analysis with Final Refined Structure
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pythainlp import word_tokenize
from pythainlp.util import normalize
import emoji
import warnings
warnings.filterwarnings('ignore')

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 5000
SELECT_K_BEST = 2000

def load_data(filepath):
    try:
        df = pd.read_excel(filepath)
        if not all(col in df.columns for col in ['paragraph', 'label']):
            raise ValueError("Excel file must contain 'paragraph' and 'label' columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = normalize(text)
    text = emoji.replace_emoji(text, replace='')
    pattern = r'[^ก-๙a-zA-Z0-9%\$฿.,+\- ]'
    text = re.sub(pattern, ' ', text)
    text = ' '.join(text.split())
    return text.strip()

def tokenize_thai_text(text):
    money_pattern = r'\d+[.,]?\d*[%\$฿]'
    money_matches = re.findall(money_pattern, text)    
    tokens = word_tokenize(text, engine="newmm")
    final_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        matched = False
        for pattern in money_matches:
            if token in pattern and i + 1 < len(tokens):
                combined = token + tokens[i+1]
                if combined in pattern:
                    final_tokens.append(combined)
                    i += 1
                    matched = True
                    break
        if not matched:
            final_tokens.append(token)
        i += 1
    return final_tokens

def preprocess_data(df):
    df['cleaned_text'] = df['paragraph'].apply(clean_text)
    df = df[df['cleaned_text'].str.len() > 0]
    print("\nClass Distribution:")
    print(df['label'].value_counts())
    return df

def fit_feature_pipeline(X_train, y_train):
    tfidf = TfidfVectorizer(
        tokenizer=tokenize_thai_text, max_features=MAX_FEATURES, ngram_range=(1, 2), lowercase=False
    )
    ch2 = SelectKBest(chi2, k=min(SELECT_K_BEST, X_train.shape[0]))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_train_selected = ch2.fit_transform(X_train_tfidf, y_train)
    return X_train_selected, tfidf, ch2

def transform_features(X_data, tfidf, ch2):
    X_tfidf = tfidf.transform(X_data)
    X_selected = ch2.transform(X_tfidf)
    return X_selected

def train_model(X_train, y_train, model_type='lr'):
    if model_type == 'lr':
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced', multi_class='multinomial')
    elif model_type == 'nb':
        model = MultinomialNB()
    elif model_type == 'svm':
        model = SVC(kernel='linear', random_state=RANDOM_STATE, class_weight='balanced', probability=True)
    else:
        raise ValueError("Invalid model type. Choose 'lr', 'nb', or 'svm'")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['neg', 'pos']))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['neg', 'pos'], yticklabels=['neg', 'pos'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    # return dictionary
    return {'accuracy': accuracy_score(y_test, y_pred), 'precision': precision_score(y_test, y_pred, average='weighted'), 'recall': recall_score(y_test, y_pred, average='weighted'), 'f1': f1_score(y_test, y_pred, average='weighted'), 'confusion_matrix': cm}

def save_model(model, tfidf, feature_selector, filename):
    model_data = {'model': model, 'tfidf': tfidf, 'feature_selector': feature_selector}
    joblib.dump(model_data, filename)
    print(f"\nModel saved as {filename}")

def load_saved_model(filename):
    try:
        return joblib.load(filename)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_new_text(model_data, text):
    if not model_data:
        return "Model not loaded correctly"
    cleaned_text = clean_text(text)
    selected_text = transform_features([cleaned_text], model_data['tfidf'], model_data['feature_selector'])
    prediction = model_data['model'].predict(selected_text)
    probabilities = model_data['model'].predict_proba(selected_text)
    return {'prediction': prediction[0], 'probabilities': {'neg': probabilities[0][0], 'pos': probabilities[0][1]}}

def main():
    #data_file = "thai_sentiment_data.xlsx"
    data_file = "gemini_mixed_sentiment_2label.xlsx"
    df = load_data(data_file)
    if df is None: return
    df = preprocess_data(df)
    
    X_raw = df['cleaned_text']
    y = df['label']
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    
    print(f"\nData split into {len(X_train_raw)} training and {len(X_test_raw)} testing samples.")
    X_train, tfidf, ch2 = fit_feature_pipeline(X_train_raw, y_train)
    X_test = transform_features(X_test_raw, tfidf, ch2)
    
    print(f"Final training features shape: {X_train.shape}")
    print(f"Final testing features shape: {X_test.shape}")
    
    model_choice = input("\nSelect model to train (lr/nb/svm): ").strip().lower()
    
    while model_choice not in ['lr', 'nb', 'svm']:
        model_choice = input("Select model to train (lr/nb/svm): ").strip().lower()
    
    print(f"\nTraining '{model_choice}' model")
    model = train_model(X_train, y_train, model_choice)
    
    test_metrics = evaluate_model(model, X_test, y_test)
    print("\nEvaluation Metrics Summary:")
    print(test_metrics)
    
    model_save_file = "thai_sentiment_model.joblib"
    save_model(model, tfidf, ch2, model_save_file)

    # --- testing with unknown label dataset
    print("\n--- Prediction Demonstration ---")
    loaded_model_data = load_saved_model(model_save_file)
    
    if loaded_model_data:
        example_texts = [
            "SJWD แย้มไตรมาส 4 สดใส มั่นใจดันรายได้ปีนี้แตะ 2.5 หมื่นล้าน",
            "เม็ดเงิน วายุภักษ์ ซื้อหุ้นค้าปลีก ดัน SET ปิดบวก 8 จุด แนวโน้มพรุ่งไซด์เวย์อัพ",
            "SABUY อ่วมหนัก Q3 ขาดทุน 865 ล้านบาท ตลท. แขวนป้าย CB มีผล 18 พ.ย.นี้",
            "เช้าวันนี้SETHD ปิดที่ระดับ 1,075.29  จุด ลดลง 7.48  จุด หรือ 0.69 %",
            "ต่างชาติ เทขาย 3.4 พันล้านบาท กด SET ปิดลบ 9 จุด"
        ]
        
        for text in example_texts:
            result = predict_new_text(loaded_model_data, text)
            print(f"\nText: '{text}'")
            print(f"  -> Prediction: {result['prediction']}")
        
            probs_formatted = ', '.join([f"{k}: {v:.2%}" for k, v in result['probabilities'].items()])
            print(f"  -> Probabilities: {probs_formatted}")

if __name__ == "__main__":
    main()