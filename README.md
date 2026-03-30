📊 Mixed Sentiment Analysis for Stock News in Thailand (Thai NLP)

พัฒนาโมเดลวิเคราะห์อารมณ์ (Sentiment Analysis) สำหรับข่าวหุ้นไทย โดยรองรับทั้ง single sentiment และ mixed sentiment เพื่อให้สะท้อนบริบทของข่าวได้แม่นยำมากขึ้น

🔧 Key Responsibilities
🔎 เก็บข้อมูลข่าวหุ้นไทยด้วย Web Scraping โดยใช้ BeautifulSoup
☁️ ใช้ Google Cloud Platform (GCP) ในการจัดการและประมวลผลข้อมูลระดับ paragraph สำหรับการ train model
🏷️ สร้าง dataset และทำการ label ข้อมูลเป็น 3 คลาส:
- Positive (pos)
- Negative (neg)
- Neutral (neu)
🧠 ออกแบบ Mixed Sentiment Patterns จำนวน 6 รูปแบบ โดยอิงหลักภาษาศาสตร์ เช่น:
- การใช้คำเชื่อม (conjunction)
- การเน้น clause แรก/หลัง
- explicit / implicit sentiment

⚙️ NLP Processing
- ตัดคำภาษาไทย (Tokenization)
- ลบ Stopwords
- แปลงข้อความเป็น Feature ด้วย TF-IDF

🤖 Model Implementation
ทดลองและเปรียบเทียบโมเดล Machine Learning และ Deep Learning ได้แก่:
- Logistic Regression (LR)
- Naïve Bayes (NB)
- Support Vector Machine (SVM)
- WangchanBERTa (Transformer-based Thai NLP model)

📈 Evaluation Metrics
ประเมินประสิทธิภาพโมเดลด้วย:
- Accuracy
- Precision
- Recall
- F1-score

🛠️ Skills
💻 Programming & Tools
- Python
- BeautifulSoup
- Google Cloud Platform (GCP)
  
🧠 Machine Learning / NLP
- Text Preprocessing (Tokenization, Stopword Removal)
- TF-IDF Feature Extraction
- Classification Models (LR, NB, SVM)
- Transformer Models (WangchanBERTa)

📊 Data & Evaluation
- Data Labeling
- Mixed Sentiment Analysis
- Model Evaluation Metrics
