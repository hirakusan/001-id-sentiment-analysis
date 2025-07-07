# Indonesian Tweet Sentiment Analysis

Proyek analisis sentimen tweet bahasa Indonesia menggunakan dataset yang telah dibersihkan dan diproses. Proyek ini mengimplementasikan berbagai teknik machine learning dan deep learning untuk klasifikasi sentimen dengan akurasi tinggi.
📊 Dataset
Source: adealvii/Cleaned-Indonesian-Tweet
Dataset ini berisi tweet bahasa Indonesia yang telah melalui proses pembersihan dan pelabelan sentimen. Dataset mencakup:

Tweet yang telah dibersihkan dari noise
Label sentimen (positif, negatif, netral)
Preprocessing yang sudah optimal untuk bahasa Indonesia

🎯 Tujuan Proyek

Menganalisis sentimen tweet berbahasa Indonesia
Membandingkan performa berbagai algoritma ML/DL
Mengimplementasikan preprocessing khusus untuk teks Indonesia
Membangun model production-ready untuk sentiment analysis

🚀 Fitur Utama
1. Data Preprocessing

Text Cleaning: Removal URL, mention, hashtag, emoji
Normalisasi: Konversi kata tidak baku ke kata baku
Tokenization: Pemisahan kata dengan handling bahasa Indonesia
Stopwords Removal: Menggunakan stopwords bahasa Indonesia
Stemming: Implementasi Sastrawi stemmer

2. Feature Engineering

TF-IDF Vectorization: Traditional feature extraction
Word2Vec: Word embeddings untuk semantic representation
FastText: Subword information untuk handling OOV
BERT Indonesian: Pre-trained transformer untuk bahasa Indonesia

3. Machine Learning Models

Naive Bayes: Baseline model
Support Vector Machine: Linear dan RBF kernel
Random Forest: Ensemble method
Logistic Regression: Linear classifier
XGBoost: Gradient boosting

4. Deep Learning Models

LSTM: Sequential modeling untuk context
BiLSTM: Bidirectional context understanding
CNN: Convolutional approach untuk text
BERT Fine-tuning: State-of-the-art transformer model

📋 Requirements
python# Core libraries
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0

# NLP libraries
nltk>=3.6.0
Sastrawi>=1.0.1
transformers>=4.15.0
torch>=1.10.0

# Indonesian NLP
indobert>=0.1.0
bahasa>=0.1.0

# Visualization
plotly>=5.0.0
wordcloud>=1.8.0

# Utility
tqdm>=4.62.0
jupyter>=1.0.0

🔧 Instalasi
Clone Repository
bashgit clone https://github.com/hirakusan/002-ml-dl-fundamental
cd indonesian-tweet-sentiment-analysis
Install Dependencies
bashpip install -r requirements.txt

Instructor: HIBA AMBARA
Email:hiba.ambara2@gmail.com
LinkedIn: https://www.linkedin.com/in/hiba-ambara-5b7213204/
telegram : https://t.me/hirakusan

📄 License
Project ini dilisensikan under MIT License - lihat file LICENSE untuk detail.
🙏 Acknowledgments

adealvii untuk dataset Cleaned-Indonesian-Tweet
Hugging Face untuk transformers library
Sastrawi untuk Indonesian NLP tools
IndonesianNLP community untuk resources


Happy Analyzing! 📊
"Understanding public sentiment through Indonesian tweets"