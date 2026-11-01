This project evaluates hallucinations in a Retrieval-Augmented
Generation (RAG) system for UAE government-service questions asked in
English, Arabic, and Arabic--English code-mixed language.
Pipeline
1. Load and validate official document passages and evaluation
   questions.
2. Normalize Arabic and English text and create development/test
   splits.
3. Retrieve evidence using hybrid BM25 + multilingual E5 dense
   retrieval, fused with Reciprocal Rank Fusion (RRF).
4. Generate answers with two systems:
   - Naive RAG: answers using retrieved context.
   - Grounded RAG: answers only from evidence and abstains when
     evidence is insufficient.
5. Apply a retrieval evidence-sufficiency gate, calibrated on the
   development split.
6. Evaluate answers using an LLM judge, multilingual NLI checks,
   citation and number-grounding checks, and reference-answer
   similarity.
7. Compare results overall and by question language, with confidence
   intervals and statistical tests.
Requirements
Install the packages used by the notebook:
pip install sentence-transformers rank_bm25 google-genai openai scikit-learn scipy matplotlib seaborn tqdm openpyxl pandas numpy transformers torch
Input data
Place these CSV files in data/raw/:
- official_document.csv (or documents.csv /
  official_documents.csv)
- final_questions.csv
The notebook expects document and question identifiers, text, language
and answerability fields, plus the reference-answer and relevance fields
used in the evaluation.
API configuration
Set the API keys GEMINI_API_KEY and GROQ_API_KEY as environment
variables or Colab Secrets. The notebook uses Gemini for answer
generation and Groq for LLM judging by default. Do not commit API keys
or .env files.
Run
Open uae_gov_rag_hallucination.ipynb in Google Colab or Jupyter.
Configure the data paths and API keys, then run the notebook cells in
order. The notebook is configured to evaluate the test split; the
development split is used to calibrate the retrieval gate threshold.
Outputs
Generated files are saved under outputs/, including:
- generations/ --- generated answers and combined results.
- tables/ --- overall and language-wise metrics, confidence
  intervals, ROUGE scores, and statistical tests.
- figures/ --- metric plots and confusion matrices.
- cache/ --- cached LLM responses.
Main evaluation measures
Hallucination rate, good-response rate, faithfulness, answer
correctness, abstention precision/recall/F1, over-abstention, citation
validity, unsupported numbers, reference similarity, and ROUGE scores.
Hallucination rate, good-response rate, faithfulness, answer
correctness, abstention precision/recall/F1, over-abstention, citation
validity, unsupported numbers, reference similarity, and ROUGE scores.
