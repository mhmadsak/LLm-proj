# Hallucination Detection Pipeline

This project implements a search-enhanced RAG pipeline for detecting hallucinated spans in multilingual LLM outputs. It retrieves external context (via Google CSE), runs a verifier model (DeepSeek), and outputs both hard- and soft-label annotations.

---

## 🚀 How to Run

```bash
# 1. Setup
git clone <your-repo-url>
cd <your-repo-name>
echo "DEEPSEEK_API_KEY=your_deepseek_api_key" >> .env
echo "GOOGLE_SEARCH_API_KEY=your_google_search_api_key" >> .env
echo "GOOGLE_SEARCH_ENGINE=your_google_cse_id" >> .env
pip install -r requirements.txt

# 2. Run inference
python inference.py \
  --input data/val/mushroom.de-val.v2.jsonl \
  --output data/preds/pred-de.jsonl \
  --hard-threshold 0.6

# 3. Evaluate
python scorer.py data/val/mushroom.de-val.v2.jsonl data/preds/pred-de.jsonl data/preds/metrics_de.txt
The results are saved in `data/preds/metrics_de.txt`.Note that this example of running is to test the evaluation on german language ,for other tests choose the desired data path