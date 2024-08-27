# **Sentiment Analysis Project**

## **Project Description**

This project aims to analyze the sentiment of product reviews using three different approaches:

* **Machine Learning (ML)** using Logistic Regression,
* **Deep Learning (DL)** using LSTM (Long Short-Term Memory),
* **Large Language Models (LLM)** using a fine-tuned BERT model from Hugging Face.

The project compares the performance of these models across multiple datasets to determine which method is the most effective for sentiment classification.

### **Objectives**

* Build models using ML, DL, and LLM to classify product reviews as positive or negative.
* Compare the models' performance based on evaluation metrics like accuracy, precision, recall, and F1-score.
* Deploy the models via a Flask API to allow real-time predictions.

## Folder Structure

```
sentiment_analysis_project/
├── data/
│   ├── dataset_1.csv
│   ├── dataset_2.csv
│   └── dataset_3.csv
├── models/
│   ├── ml_model.pkl
│   ├── dl_model.h5
│   └── llm_model.pt
├── notebooks/
│   └── Sentiment_Analysis.ipynb
├── src/
│   ├── ml_model.py
│   ├── dl_model.py
│   └── llm_model.py
├── results/
│   └── comparison_results.csv
├── app.py
├── requirements.txt
└── README.md
```

## **Datasets**

* The `data/` folder contains three datasets in `.csv` format, each consisting of two columns:
  * **`review`** : The text of the product review.
  * **`sentiment`** : The label indicating positive (1) or negative (0) sentiment.

## **Setup & Installation**

1. **Clone the repository:**

   ```
   git clone <your-repo-url>
   cd sentiment_analysis_project

   ```
2. **Install dependencies:**
   Install the required libraries by running the following command:

   ```
   pip install -r requirements.txt

   ```

* **Prepare datasets:**
  * Place your CSV datasets inside the `data/` folder.
  * Ensure that the datasets are formatted with two columns: `review` and `sentiment`.
* **Running Models:**
  * Navigate to the `src/` directory and run the specific model scripts:

---

## **API Endpoints**

* **ML Model Prediction (Logistic Regression):**
  * Endpoint: `/predict/ml`
  * Method: `POST`
  * Payload:
    <pre><div class="dark bg-gray-950 rounded-md border-[0.5px] border-token-border-medium"><div class="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>json</span><div class="flex items-center"><span class="" data-state="closed"><button class="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" class="icon-sm"><path fill="currentColor" fill-rule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clip-rule="evenodd"></path></svg>Copy code</button></span></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-json">{
      "text": "This product is amazing!"
    }
    </code></div></div></pre>
* **DL Model Prediction (LSTM):**
  * Endpoint: `/predict/dl`
  * Method: `POST`
  * Payload:
    <pre><div class="dark bg-gray-950 rounded-md border-[0.5px] border-token-border-medium"><div class="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>json</span><div class="flex items-center"><span class="" data-state="closed"><button class="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" class="icon-sm"><path fill="currentColor" fill-rule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clip-rule="evenodd"></path></svg>Copy code</button></span></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-json">{
      "text": "Terrible quality, would not buy again."
    }
    </code></div></div></pre>
* **LLM Model Prediction (BERT):**
  * Endpoint: `/predict/llm`
  * Method: `POST`
  * Payload:
    <pre><div class="dark bg-gray-950 rounded-md border-[0.5px] border-token-border-medium"><div class="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>json</span><div class="flex items-center"><span class="" data-state="closed"><button class="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" class="icon-sm"><path fill="currentColor" fill-rule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clip-rule="evenodd"></path></svg>Copy code</button></span></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-json">{
      "text": "The performance is outstanding!"
    }
    </code></div></div></pre>

Each endpoint will return a JSON response with the predicted sentiment (1 for positive and 0 for negative)
