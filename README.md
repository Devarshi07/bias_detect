# Hotel Bias Detection - Quick Guide

## 📁 Folder Structure

BIAS/
      
    ├── .env
    ├── requirements.txt
    ├── processed_boston_reviews.csv              # Input: Hotel reviews
    ├── hotel_bias_detection.py                   # Core: Bias detection logic
    │
    ├── generate_response/
    │   └── generate_chatbot_response.py     # Step 1: Generate AI summaries
    │
    ├── pipeline/
    │   └── stage_bias_detection.py              # Step 2: Run bias detection
    │
    ├── response/
    │   ├── chatbot_responses.csv                # Generated AI responses (CSV)
    │   └── chatbot_responses.parquet            # Generated AI responses (Parquet)
    │
    └── evaluation/results
        ├── hotel-bias-scores.json               # Detailed bias data per hotel
        ├── hotel-bias-results.csv               # Results table (Excel-friendly)
        ├── hotel-bias-results.parquet           # Results table (binary)
        └── bias-detection-summary.json          # Overall statistics
    
---

## 🚀 Quick Start

### 1. Setup
#### Install dependencies
pip install -r requirements.txt

#### Add your API key to .env
echo "XAI_API_KEY=xai-your_key_here" > .env

### 2. Generate AI Responses
python generate_response/generate_chatbot_response.py

Output: response/chatbot_responses.parquet

### 3. Run Bias Detection
python pipeline/stage_bias_detection.py

Output: 4 files in evaluation/results/

### 4. View Results
#### Summary
cat evaluation/results/bias-detection-summary.json

#### Detailed
open evaluation/results/hotel-bias-results.csv

---

## 📋 What Each File Does

| File                           | Purpose                                | When to Use                     |
|-------------------------------|--------------------------------------------|----------------------------------|
| `generate_chatbot_response_xai.py` | Creates AI summaries of hotel reviews       | Run first — once per dataset     |
| `stage_bias_detection.py`     | Checks AI responses for potential bias     | Run second — analyzes responses  |
| `hotel_bias_detection.py`     | Core bias-detection logic used by pipeline | Auto-imported by other scripts   |
| `chatbot_responses.parquet`   | Stores AI-generated review summaries       | Generated during Step 1          |
| `hotel-bias-results.csv`      | Outputs full bias-detection results        | Open in Excel or Sheets          |
| `bias-detection-summary.json` | Provides high-level statistics overview    | For quick health/status checks   |


---

## 🎯 What Gets Detected
3 Types of Bias:
- Over-reliance on Negative - AI too negative when reviews are mixed
- Missing Data Acknowledgment - AI doesn't mention limited reviews (<4)
- Rating Disparity - AI negative despite 4+ star ratings

---

## ⚙️ Configuration

Change number of hotels:
```
# In generate_chatbot_response_xai.py
NUM_HOTELS = 5  # Change to 10, 20, etc.
```

Adjust bias thresholds:
```
# In hotel_bias_detection.py

self.config = {
    'neg_sentiment_threshold': 0.7,        # 70% negativity = bias
    'min_reviews_threshold': 4,            # <4 reviews = sparse data
    'rating_disparity_threshold': 4.0      # 4+ stars = good rating
}
```
---

## 📝 requirements.txt
```
pandas>=2.0.0
pyarrow>=12.0.0
python-dotenv>=1.0.0
openai>=1.0.0
```


