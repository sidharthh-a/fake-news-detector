# 📰 Fake News Detector for Students

An AI-powered Fake News Detection system built using DistilBERT and deployed via Streamlit Cloud.  
The model classifies news articles as **Real** or **Fake** based on learned linguistic patterns.

---

## 🚀 Live Demo

🔗 https://fake-news-detector-jfiyyappx5hkqqjewkem6rk.streamlit.app/

---

## 🤗 Hugging Face Model

🔗 https://huggingface.co/sidharth-ai/fake-news-detector

---

## 🧠 Project Overview

This project implements a transformer-based NLP classifier to detect fake news using a fine-tuned **DistilBERT** model.

The system was trained on the ISOT Fake and True News Dataset and optimized to reduce overfitting using:

- Stratified train-test split  
- Backbone freezing  
- Regularization (weight decay)  
- Controlled training epochs  

The final model was deployed using:

- Hugging Face Model Hub (for model hosting)
- Streamlit Cloud (for web interface deployment)

---

## 📊 Model Performance

- **Validation Accuracy:** 97.6%  
- **F1 Score:** 97.5%  
- **Precision:** 96.1%  
- **Recall:** 98.9%  

These results indicate strong classification performance while avoiding overfitting issues observed during initial experiments.

---

## 🛠 Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Streamlit
- Hugging Face Hub
- Scikit-learn

---

## ⚙️ Deployment Architecture

User → Streamlit Cloud → Hugging Face Model Hub → DistilBERT Inference

The model is hosted separately from the application, ensuring lightweight GitHub deployment and scalable inference.

---

## ⚠️ Limitations

- The model performs **linguistic pattern classification**, not real-time fact verification.
- It does not access live databases or official records.
- Performance is influenced by training dataset characteristics (ISOT dataset bias).
- Highly well-written misinformation may be misclassified.

This project demonstrates ML classification capabilities, not full fact-checking functionality.

---

## 📌 Future Improvements

- Integration with NewsAPI for real-time cross-checking  
- Explainable AI features (attention visualization)  
- Multi-dataset training for better generalization  
- Confidence threshold calibration  

---

## 👨‍💻 Author

**Sidharth**  
Machine Learning & NLP Enthusiast  
