# Cascade Intelligence: 3-Tier Classification + RAG-Grounded Response Generation

An end-to-end support ticket classification and response generation system combining a 3-tier confidence-gated classification cascade with retrieval-augmented generation (RAG). The system classifies tickets into categories and priorities, retrieves relevant knowledge base articles, and generates grounded draft responses with citations — all without hallucinating when the KB has no answer.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Pipeline Stages in Detail](#pipeline-stages-in-detail)
3. [Performance: Our Pipeline vs. Naive Baseline](#performance-our-pipeline-vs-naive-baseline)
4. [Classification Results](#classification-results)
5. [Confidence Calibration](#confidence-calibration)
6. [Open-Source Models and APIs](#open-source-models-and-apis)
7. [Cost Analysis](#cost-analysis)
8. [Setup and Installation](#setup-and-installation)
9. [How to Reproduce Results](#how-to-reproduce-results)
10. [Latency / Cost / Accuracy Trade-offs](#latency--cost--accuracy-trade-offs)
11. [Limitations](#limitations)
12. [Monitoring Plan for Production](#monitoring-plan-for-production)
13. [Project Structure](#project-structure)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Raw Support Ticket                              │
└──────────────────────────┬───────────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  PII Scrubber  (spaCy NER + regex)                                       │
│  Redacts: emails, phones, SSNs, IPs, credit cards, names, orgs, locs    │
│  Then: lowercasing → lemmatization → placeholder stripping               │
└──────────────────────────┬───────────────────────────────────────────────┘
                           ▼
          ┌─────── CLASSIFICATION CASCADE ────────┐
          │                                       │
          ▼                                       │
┌───────────────────────┐                         │
│  Tier 1: Rules +      │  conf ≥ 0.90            │
│  TF-IDF + LogReg      │──────────────────┐      │
│  (local, ~2ms)        │                  │      │
└─────────┬─────────────┘                  │      │
          │ conf < 0.90                    │      │
          ▼                                │      │
┌───────────────────────┐                  │      │
│  Tier 2: SetFit +     │  conf ≥ 0.65    │      │
│  Temperature Calib.   │──────────────────┤      │
│  (local, ~50ms)       │                  │      │
└─────────┬─────────────┘                  │      │
          │ conf < 0.65                    │      │
          ▼                                │      │
┌───────────────────────┐                  │      │
│  Tier 3: Groq LLM     │                 │      │
│  qwen/qwen3-32b       │─────────────────┤      │
│  (API call, ~1-3s)    │                  │      │
└───────────────────────┘                  │      │
                                           ▼      │
                                 ┌─────────────┐  │
                                 │  Operating   │  │
                                 │  Point Router│  │
                                 └──────┬──────┘  │
                                        │         │
          ┌─────────────────────────────┘         │
          │                                       │
          ▼                                       │
┌──────────────────── RAG PIPELINE ───────────────┘
│
│  1. HyDE Query Expansion (Groq qwen3-32b)
│     └─ Generates 2 hypothetical answers → 3 total queries
│
│  2. Dual Retrieval
│     ├─ BM25 (sparse keyword matching)
│     └─ FAISS (dense semantic search, all-MiniLM-L6-v2)
│
│  3. Reciprocal Rank Fusion (k=60)
│     └─ Merges BM25 + FAISS rankings into unified list
│
│  4. Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
│     └─ Reranks top 30 → returns top 3 chunks
│
│  5. Abstention Gate (threshold = 0.35)
│     └─ If max rerank score < 0.35 → abstain (no hallucinated answer)
│
│  6. Response Generation (Groq qwen3-32b)
│     └─ Grounded JSON: {"draft_response": "...", "citations": [0, 1]}
│
└──────────────────────────────────────────────────
          │
          ▼
┌──────────────────────────────────────────────────┐
│  Final Output                                     │
│  ticket_id, predicted_category, predicted_priority│
│  confidence_score, tier_used, routing_action,     │
│  abstain_flag, top_3_sources, draft_response,     │
│  citations, retrieval_max_score                   │
└──────────────────────────────────────────────────┘
```

---

## Pipeline Stages in Detail

### Stage 1: PII Scrubbing (`backend/src/Scrubber/scrub.py`)

Protects personally identifiable information before any model sees the text.

| PII Type | Detection Method | Replacement |
|----------|-----------------|-------------|
| Email | Regex | `[REDACTED_EMAIL]` |
| Phone | Regex | `[REDACTED_PHONE]` |
| SSN | Regex | `[REDACTED_SSN]` |
| Credit Card | Regex | `[REDACTED_CC]` |
| IP Address | Regex | `[REDACTED_IP]` |
| ZIP Code | Regex | `[REDACTED_ZIP]` |
| Person Name | spaCy NER | `[REDACTED_PERSON]` |
| Organization | spaCy NER | `[REDACTED_ORG]` |
| Location | spaCy NER | `[REDACTED_LOCATION]` |

After redaction: `lowercase → lemmatize (spaCy) → strip synthetic placeholders`.

**Example:**
```
Input:  "Hi, my name is Sarah Johnson, email sarah.j@outlook.com,
         card 4532 1234 5678 9012 was charged twice."
Output: "hi my name be [REDACTED_PERSON] email [REDACTED_EMAIL]
         card [REDACTED_CC] be charge twice"
```

### Stage 2: Tier 1 Classification (`backend/src/Classification/tier_1.py`)

A hybrid of data-driven keyword rules and logistic regression.

- **DataDrivenRules**: Extracts top-10 TF-IDF discriminative n-grams per class. Matches via regex word boundaries.
- **CalibratedLogReg**: TF-IDF (word 1-2 grams + char 3-5 grams) → LogisticRegression (C=5.0, balanced class weights) → Isotonic calibration via 5-fold StratifiedKFold.
- **LabelHead**: Runs both, picks the one with higher confidence.
- **Threshold**: `conf ≥ 0.90` → auto-route. Otherwise escalate to Tier 2.

### Stage 3: Tier 2 Classification (`backend/src/Classification/tier_2.py`)

SetFit (Sentence Transformer Fine-Tuning) few-shot classifiers with temperature calibration.

- **Models**: Two separate SetFit models (category + priority), fine-tuned via contrastive learning on `sentence-transformers/all-MiniLM-L6-v2`.
- **Calibration**: Temperature scaling with T=1.5, applied to logits before softmax. Isotonic regression disabled — it overfit to the perfect validation set and collapsed all confidence scores to 1.0.
- **Threshold**: `conf ≥ 0.65` → auto-route. Otherwise escalate to Tier 3.

> **Why 0.65?** With T=1.5 calibration, clear tickets score 0.92–0.97. Genuinely ambiguous tickets (e.g., "cancel + refund" mixed) score 0.60–0.68 and correctly fall through to Tier 3 for LLM resolution.

### Stage 4: Tier 3 LLM Classification (`backend/src/Classification/tier_3.py`)

Final arbitration via Groq-hosted LLM.

- **Model**: `qwen/qwen3-32b` via Groq API
- **Prompt**: Includes ticket text, KB excerpt (truncated to 3000 chars), and explicit valid label lists:
  - Categories: `Billing inquiry`, `Cancellation request`, `Product inquiry`, `Refund request`, `Technical issue`
  - Priorities: `Critical`, `High`, `Medium`, `Low`
- **Output**: Structured JSON validated against `TicketPrediction` Pydantic schema
- **Retries**: Up to 2 retries with backoff. Falls back across model candidates on rate-limit.
- **`<think>` stripping**: qwen3 emits `<think>...</think>` reasoning blocks stripped before JSON parsing.

### Stage 5: Operating Point Router (`backend/src/Classification/pipeline.py`)

Maps confidence scores to routing decisions:

| Confidence Range | Routing Action | Meaning |
|-----------------|----------------|---------|
| ≥ 0.90 | `auto_route` | High confidence — route automatically |
| 0.65 – 0.89 | `suggest` | Moderate confidence — show to agent for confirmation |
| < 0.65 | `manual_review` | Low confidence — human must decide |

### Stage 6: HyDE Query Expansion (`backend/src/RAG/HyDe.py`)

Hypothetical Document Embeddings — generates synthetic answers to improve retrieval recall.

- Sends the ticket to Groq and asks for 2 hypothetical help-desk responses
- Returns 3 queries: `[original_ticket, hypothesis_1, hypothesis_2]`
- Hypothetical answers are semantically closer to KB chunks than raw questions — bridging the query-document vocabulary gap

### Stage 7: Two-Stage Retrieval (`backend/src/RAG/retrieval/`)

**Stage 1 — Recall** (per query, per retriever):
- **BM25** (`bm25_retriever.py`): Sparse keyword matching via `rank_bm25`. Good for exact terms like "refund", "password reset". Returns top 15 chunks per query.
- **FAISS** (`faiss_retriever.py`): Dense semantic search via `all-MiniLM-L6-v2` (384-dim, cosine similarity via IndexFlatIP on normalized vectors). Good for paraphrases. Returns top 15 chunks per query.

With 3 HyDE queries × 2 retrievers = 6 ranked lists.

**Fusion** (`rrf.py`): Reciprocal Rank Fusion with k=60:
```
score(chunk) = Σ 1/(k + rank_in_list)
```
Returns top 30 fused candidates.

**Stage 2 — Reranking** (`reranker.py`): Cross-encoder `ms-marco-MiniLM-L-6-v2` scores each `(query, chunk)` pair jointly. Returns top 3 chunks. Scores are raw logits — positive means relevant, negative means irrelevant.

### Stage 8: Abstention Gate (`backend/src/generation/abstention_gate.py`)

- **Threshold**: 0.35 on the maximum cross-encoder rerank score
- `max(rerank_scores) ≥ 0.35` → generate response
- `max(rerank_scores) < 0.35` → abstain with: *"Insufficient information in the knowledge base to answer this question. Please escalate to a human agent."*

Prevents hallucinated answers when the KB has no relevant content.

### Stage 9: Response Generation (`backend/src/generation/generator.py`)

- **Model**: `qwen/qwen3-32b` via Groq, temperature 0.2, max 1024 tokens
- **Prompt**: Numbered source blocks + ticket text + strict JSON schema instruction
- **Output**: `{"draft_response": "...", "citations": [0, 1, 2]}`
- **Citation-aware retry**: If the model returns no citations, retries up to 2 times with a reinforced prompt
- **`<think>` stripping**: Same qwen3 reasoning block handling as Tier 3

---

## Performance: Our Pipeline vs. Naive Baseline

### What the Naive Baseline Does

The `naive_rag/` folder contains a minimal reference implementation:
- Single TF-IDF + LogReg (no cascade, no confidence gating)
- Raw query → single FAISS cosine search → direct Groq call
- No HyDE, no BM25, no cross-encoder reranking, no abstention gate

### Head-to-Head Comparison

| Dimension | Naive Baseline | Our Pipeline |
|-----------|---------------|--------------|
| **Classification** | Single LogReg, always runs | 3-tier cascade — cheap tiers handle easy tickets |
| **Calibration** | None | Temperature scaling (T=1.5), honest confidence |
| **Confidence routing** | None — always auto-route everything | `auto_route` / `suggest` / `manual_review` based on actual uncertainty |
| **Query expansion** | Raw ticket text only | HyDE: 2 hypothetical answers expand the search space |
| **Retrieval** | FAISS only (cosine similarity) | BM25 + FAISS dual retrieval with RRF fusion |
| **Reranking** | None | Cross-encoder ms-marco-MiniLM-L-6-v2 |
| **Hallucination prevention** | None — always generates an answer | Abstention gate refuses to answer when KB score < 0.35 |
| **Citation enforcement** | None | Forced JSON with mandatory citation indices + retry |
| **PII protection** | None | Regex + NER scrubbing before any model sees the text |
| **Cost per ticket** | Always hits LLM | Tier 1/2 catch easy tickets for $0 |

### RAG Quality on 5 Test Queries (measured live)

| Query | Naive RAG | Our RAG | Winner |
|-------|-----------|---------|--------|
| "What is your refund policy?" | Answers (no source enforcement) | Answers with citations from `refund_policy` (score 6.34) | **Ours** |
| "I was charged twice, explain billing" | Answers (ungrounded) | Answers from `billing_policy` (score 4.81) | **Ours** |
| "How do I cancel my subscription?" | Answers (ungrounded) | Answers from `cancellation_policy` (score 7.44) | **Ours** |
| "File upload crashes with ERR_UPLOAD_TIMEOUT" | **Hallucinated** a plausible answer | Abstains correctly (score -8.0) | **Ours** |
| "Do you have a Slack integration?" | **Hallucinated** "Yes, we support Slack" | Abstains correctly (score -11.4) | **Ours** |

The critical difference: Naive RAG **fabricates answers** for out-of-KB questions. Our pipeline abstains and routes to a human agent.

### Latency Comparison (8 queries, measured)

| Metric | Naive RAG | Our Pipeline |
|--------|-----------|--------------|
| Min latency | 4,493 ms | 790 ms (KB hit, Tier 2) |
| Max latency | 6,577 ms | 20,189 ms (Tier 3 + HyDE + generation) |
| Consistency | High (always 1 Groq call) | Variable (0–3 Groq calls depending on tier) |

Our pipeline is faster on easy tickets (KB hit, Tier 1/2) and slower on hard ones (Tier 3 + full RAG). The naive baseline always pays the same flat cost.

---

## Classification Results

### Model Performance (Trained on 3,388 examples, validated on 1,130)

| Model | Category Macro-F1 | Priority Macro-F1 |
|-------|-------------------|-------------------|
| Tier 1 — TF-IDF + LogReg | **1.00** | **0.93** |
| Tier 2 — SetFit | **1.00** | **0.98** |

### Real-World Adversarial Test (12 tough tickets, end-to-end)

| Ticket Type | Result | Notes |
|-------------|--------|-------|
| Clear billing duplicate charge | ✓ Billing inquiry (0.947) | auto_route |
| Clear technical — 500 error | ✓ Technical issue (0.906) | auto_route |
| Clear cancellation | ✓ Cancellation request (0.921) | auto_route |
| Clear refund — unopened item | ✓ Refund request (0.903) | auto_route |
| Clear product — enterprise SLA | ✓ Product inquiry (0.980) | auto_route |
| Sarcastic "yet another charge" | ✓ Billing inquiry (0.790) | suggest |
| Spanish cancel + refund | ✓ Cancellation request (0.909) | auto_route |
| Ambiguous cancel + refund | ✓ **Tier 3** → Cancellation request (0.920) | manual_review |
| Vague anger "nothing works" | ✗ Refund request (0.701) | should be Technical; routes to `suggest` |
| Fraud / account lock | ✗ Refund request (0.933) | "unauthorised purchases" reads as refund |
| Single word "Urgent" | ✓ **Tier 3** triggered → manual_review | |
| Pure symbols "???" | ✓ **Tier 3** triggered → manual_review | |

**Accuracy on labelled tickets: 7/9 = 78%**

Misclassifications are caught by the routing system: the vague anger ticket gets `suggest` (agent reviews), not `auto_route`. Only the fraud case (0.933 confidence) auto-routes incorrectly — it needs dedicated training examples for the account-compromise pattern.

### Tier Distribution on Clear Tickets

| Tier | Triggers when | Typical confidence |
|------|--------------|-------------------|
| Tier 1 (LogReg) | Unambiguous, keyword-rich text | 0.90–0.98 |
| Tier 2 (SetFit) | Nuanced language, sentence-level semantics needed | 0.65–0.97 |
| Tier 3 (Groq) | Confidence < 0.65 or genuinely ambiguous mixed-intent | 0.30–0.95 |

---

## Confidence Calibration

### The Problem We Fixed

The SetFit models were trained to F1=1.0 on the validation set. During calibration, temperature scaling minimizes NLL — when every prediction is already correct, this drives the temperature toward zero (making the model maximally sharp). The original temperature was **T=0.14**, which collapsed all confidence scores to ≥ 0.997 regardless of actual certainty.

```
Before (T=0.14):
  "cancel + get refund"     → Billing inquiry   conf=1.000  ← overconfident, wrong
  "please cancel my plan"   → Cancellation      conf=1.000  ← overconfident, happens to be right
  "help"                    → Cancellation      conf=0.739  ← still too high

After (T=1.5, isotonic disabled):
  "cancel + get refund"     → Billing inquiry   conf=0.603  → manual_review (Tier 3 resolves)
  "please cancel my plan"   → Cancellation      conf=0.921  → auto_route ✓
  "help"                    → Cancellation      conf=0.295  → Tier 3 ✓
```

### Current Configuration

| Parameter | Value | Effect |
|-----------|-------|--------|
| Category temperature (T) | **1.5** | Softens the distribution; honest uncertainty for ambiguous tickets |
| Isotonic regression | **Disabled** | Was overfit to 100%-accurate val set; mapped everything to 1.0 |
| Tier 2 gate threshold | **0.65** | Raised from 0.45 so genuinely uncertain tickets reach Tier 3 |

### Abstention Logic

**Classification cascade:**
```
Tier 1: conf < 0.90 → escalate to Tier 2
Tier 2: conf < 0.65 → escalate to Tier 3
Tier 3: always returns a prediction (final fallback)
```

**Operating point router (post-classification):**
```
conf ≥ 0.90 → auto_route    (act on it automatically)
conf 0.65–0.89 → suggest    (show to agent for confirmation)
conf < 0.65 → manual_review (human must decide)
```

**RAG abstention gate:**
```
max cross-encoder score ≥ 0.35 → generate grounded response
max cross-encoder score < 0.35 → abstain (no KB answer)
```

---

## Open-Source Models and APIs

### Models (all open-source, free to use)

| Model | Type | Where Used | Size | License |
|-------|------|-----------|------|---------|
| `sentence-transformers/all-MiniLM-L6-v2` | Bi-encoder | FAISS embeddings, SetFit backbone | 22M params | Apache 2.0 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder | Reranking | 22M params | Apache 2.0 |
| `qwen/qwen3-32b` | Causal LLM | Tier 3 classification, HyDE, generation | 32B params | Apache 2.0 |
| `en_core_web_sm` | NER/NLP | PII scrubbing, lemmatization | 12M params | MIT |

### APIs

| API | Provider | Purpose | Pricing |
|-----|----------|---------|---------|
| Groq Cloud API | Groq | Hosts qwen/qwen3-32b for Tier 3, HyDE, and generation | Free tier available |

### Libraries

| Library | Purpose |
|---------|---------|
| `scikit-learn` | TF-IDF vectorization, LogisticRegression, calibration |
| `setfit` | Few-shot text classification via contrastive learning |
| `sentence-transformers` | Bi-encoder embeddings and cross-encoder reranking |
| `faiss-cpu` | Approximate nearest neighbor search |
| `rank-bm25` | BM25 sparse retrieval |
| `spacy` | NER-based PII detection and lemmatization |
| `groq` | Python client for Groq Cloud API |
| `flask` | Backend REST API server |
| `pydantic` | Schema validation for all structured outputs |

---

## Cost Analysis

### Per-Ticket Cost Breakdown

| Component | Tokens (approx.) | Cost (Groq free tier) | Cost (Groq paid) |
|-----------|-------------------|----------------------|-------------------|
| **Tier 1 (LogReg)** | 0 | $0.00 | $0.00 |
| **Tier 2 (SetFit)** | 0 | $0.00 | $0.00 |
| **Tier 3 (Groq LLM)** | ~1,500 input + ~200 output | $0.00 (free tier) | ~$0.0002 |
| **HyDE expansion** | ~200 input + ~500 output | $0.00 (free tier) | ~$0.0001 |
| **Response generation** | ~1,500 input + ~300 output | $0.00 (free tier) | ~$0.0003 |

**Best case** (Tier 1 catches it): **$0.00** API cost — only RAG generation hits the API.

**Worst case** (falls to Tier 3): ~4,200 total tokens across 3 API calls → **~$0.0006** per ticket on paid plans.

### Compute Costs

| Resource | Cost |
|----------|------|
| Kaggle GPU (model training) | Free (Kaggle notebooks) |
| Local inference (Tiers 1-2, retrieval, reranking) | CPU only, no GPU needed |
| Groq API (Tier 3, HyDE, generation) | Free tier |
| **Total project cost** | **$0.00** |

---

## Setup and Installation

### Prerequisites

- Python 3.10+
- A Groq API key (free at [console.groq.com](https://console.groq.com))

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd ticket-classifier

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Set your Groq API key
echo 'GROQ_API_KEY=gsk_your_key_here' > backend/.env

# 6. Start the server
python3 backend/app.py
# → Server runs at http://127.0.0.1:5001
```

### Pre-trained Model Artifacts

The following artifacts must exist in `backend/results/` (produced by the Kaggle training notebook):

```
backend/results/
├── training_manifest.json          # Model paths, thresholds, metrics
├── models/
│   ├── baseline_cat_pipeline.pkl   # Tier 1 category (LabelHead)
│   ├── baseline_pri_pipeline.pkl   # Tier 1 priority (LabelHead)
│   ├── setfit_category/            # Tier 2 category (SetFit)
│   ├── setfit_priority/            # Tier 2 priority (SetFit)
│   ├── calibration_category.pkl    # Temperature calibrator (T=1.5)
│   ├── calibration_priority.pkl    # Temperature calibrator (T=1.5)
│   ├── bm25_index.pkl              # BM25 sparse index
│   ├── faiss_index.bin             # FAISS dense index
│   └── chunk_embeddings.npy        # Precomputed KB embeddings
└── processed/
    └── kb_chunks.jsonl             # 117 chunked KB entries
```

The server dynamically picks the most recently modified `training_manifest.json` under `backend/`, so a new training run is automatically picked up on the next restart.

---

## How to Reproduce Results

### 1. Train Models (Kaggle)

Open `model_training2.py` as a Kaggle notebook with the following datasets attached:
- `suraj520/customer-support-ticket-dataset` — 8,469 labeled support tickets
- `saadmakhdoom/ecommerce-faq-chatbot-dataset` — FAQ data for KB

Or use the synthetic dataset (`customer_support_5k.csv`, 5,648 tickets) generated by `generate_dataset.py`.

Run all cells. Download the `results/` folder from `/kaggle/working/` and place it at `backend/results/`.

### 2. Recalibrate (if needed)

If SetFit confidence looks overconfident (scores clustering near 1.0), patch the calibration:

```python
import joblib

for path in ["backend/results/models/calibration_category.pkl",
             "backend/results/models/calibration_priority.pkl"]:
    pkl = joblib.load(path)
    pkl["temperature"] = 1.5      # soften — was 0.14 after training on perfect val set
    pkl["use_isotonic"] = False   # disable overfit isotonic calibrators
    joblib.dump(pkl, path)
```

### 3. Run the Pipeline

```bash
python3 backend/app.py
```

Test via curl:
```bash
curl -X POST http://127.0.0.1:5001/predict_text \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice for my subscription this month"}'
```

Example response:
```json
{
  "prediction": {
    "predicted_category": "Billing inquiry",
    "predicted_priority": "Medium",
    "confidence_score": 0.947,
    "tier_used": "setfit",
    "routing_action": "auto_route",
    "abstain_flag": false,
    "draft_response": "If you were charged twice, please contact our billing support team...",
    "citations": "0",
    "top_3_retrieved_sources": "billing_policy|billing_policy|cancellation_policy",
    "retrieval_max_score": 4.247
  }
}
```

Or open http://127.0.0.1:5001 in a browser for the web UI.

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## Latency / Cost / Accuracy Trade-offs

### Per-Stage Analysis

| Stage | Latency | API Cost | Notes |
|-------|---------|----------|-------|
| **PII Scrubber** | ~10ms | $0 | Removes signal (names, orgs) but prevents PII leaks |
| **Tier 1 (LogReg)** | ~2ms | $0 | Fast filter — catches unambiguous keyword-rich tickets |
| **Tier 2 (SetFit)** | ~50ms | $0 | Sentence-level embeddings; catches nuanced language |
| **Tier 3 (Groq LLM)** | ~1–3s | ~$0.0002 | Used only for low-confidence or genuinely ambiguous tickets |
| **HyDE** | ~1–2s | ~$0.0001 | Improves retrieval recall significantly |
| **BM25 Retrieval** | ~1ms | $0 | Complements FAISS for exact keyword matches |
| **FAISS Retrieval** | ~5ms | $0 | Dense search for paraphrases |
| **RRF Fusion** | <1ms | $0 | Pure computation |
| **Cross-Encoder Rerank** | ~100ms | $0 | High precision on top 30 candidates |
| **Abstention Gate** | <1ms | $0 | Prevents hallucination — biased toward refusing |
| **Generation** | ~1–3s | ~$0.0003 | Most visible component; citation enforcement adds reliability |

### End-to-End Latency Budget

| Scenario | Total Latency | API Calls |
|----------|--------------|-----------|
| Tier 1 catches it (KB hit) | ~3–5s | 2 (HyDE + generation) |
| Tier 2 handles it (KB hit) | ~3–5s | 2 (HyDE + generation) |
| Falls to Tier 3 (KB hit) | ~6–9s | 3 (Tier 3 + HyDE + generation) |
| Any tier (KB miss, abstains) | ~2–4s | 1 (HyDE only — generation skipped) |

The bottleneck is always the Groq API (~1–3s per call), not local inference.

---

## Limitations

1. **Sentence encoder OOD blindness**: `all-MiniLM-L6-v2` maps any token sequence — including gibberish, emojis, or random strings — into the nearest embedding cluster with high confidence. There is no "unknown" region. An energy-based or Mahalanobis OOD detector would catch these.

2. **Fraud / account compromise misclassification**: Tickets saying "unauthorised purchases on my account" overlap heavily with refund training language. The model needs dedicated training examples for the account-fraud pattern to distinguish it from Billing inquiry.

3. **Single domain KB**: The KB covers e-commerce and SaaS support only. Tickets about other domains will trigger abstention on the RAG side (correctly) but may be misclassified on the classification side.

4. **English-primary**: All models (spaCy, SetFit, cross-encoder, qwen3) are English-focused. Spanish text happens to work via embedding space proximity but is not guaranteed.

5. **No conversation context**: Each ticket is classified independently. Multi-turn conversations are not linked.

6. **Groq rate limits**: Free tier has per-minute token limits. Batch processing at scale requires a paid plan or rate limiting.

7. **KB staleness**: The knowledge base is static. New products, policies, or procedures require manual KB updates and re-indexing (`faiss_index.bin`, `bm25_index.pkl`, `kb_chunks.jsonl`).

8. **sklearn version coupling**: Tier 1 models are pickled sklearn objects — upgrading scikit-learn can break deserialization. Version pinned to 1.6.1.

---

## Monitoring Plan for Production

### Classification Health

| Metric | Alert Threshold | Why |
|--------|-----------------|-----|
| Tier 3 rate | > 40% for 1 hour | Tier 1/2 models degraded or new ticket types emerging |
| Mean confidence | Drops > 10% week-over-week | Data drift |
| `manual_review` rate | > 50% | System is uncertain — investigate threshold or model drift |

### RAG Health

| Metric | Alert Threshold | Why |
|--------|-----------------|-----|
| Mean retrieval max score | < 0.3 for 1 hour | KB coverage dropping |
| Generation abstain rate | > 30% | KB needs expansion for current ticket topics |
| Citation count per response | Mean < 0.5 | Generator may be ignoring KB sources |

### Latency and Cost

| Metric | Alert Threshold | Why |
|--------|-----------------|-----|
| P95 latency | > 10s | Groq API slow or rate-limited |
| Groq API error rate | > 5% | Rate limit, key issue, or model deprecation |
| Daily API cost | > budget cap | Increase Tier 1/2 catch rate or add caching |

### Recommended Feedback Loop

```
1. Log every prediction with unique request_id
2. Expose /feedback endpoint: {request_id, correct_category, correct_priority, quality: 1-5}
3. Weekly: compute accuracy on feedback-labeled subset
4. Monthly: retrain Tier 1/2 on accumulated feedback
5. Quarterly: expand KB with new FAQ patterns from ticket logs
```

---

## Project Structure

```
ticket-classifier/
├── backend/
│   ├── app.py                              # Flask REST API (port 5001)
│   ├── Data/Raw_Data/kb.md                 # Knowledge base (31K chars, 117 chunks)
│   ├── results/
│   │   ├── training_manifest.json          # Model paths, thresholds, metrics
│   │   ├── models/                         # All serialized model artifacts
│   │   └── processed/kb_chunks.jsonl       # 117 chunked KB entries
│   └── src/
│       ├── Classification/
│       │   ├── pipeline.py                 # SupportCascadePipeline orchestrator
│       │   ├── tier_1.py                   # Rules + LogReg (local, ~2ms)
│       │   ├── tier_2.py                   # SetFit + calibration (local, ~50ms)
│       │   └── tier_3.py                   # Groq LLM arbitration (~1-3s)
│       ├── RAG/
│       │   ├── HyDe.py                     # Hypothetical Document Embeddings
│       │   └── retrieval/
│       │       ├── bm25_retriever.py       # Sparse keyword retrieval
│       │       ├── faiss_retriever.py      # Dense semantic retrieval
│       │       ├── reranker.py             # Cross-encoder reranking
│       │       ├── rrf.py                  # Reciprocal Rank Fusion
│       │       └── two_stage_retriever.py  # Orchestrates retrieval pipeline
│       ├── generation/
│       │   ├── abstention_gate.py          # Refuse if KB score < 0.35
│       │   ├── generator.py                # Groq response generation + retry
│       │   ├── prompts.py                  # Prompt templates
│       │   └── schemas.py                  # Pydantic output schemas
│       └── Scrubber/
│           ├── scrub.py                    # PII scrubbing + normalization
│           └── Rules.py                    # TicketPrediction Pydantic schema
├── naive_rag/                              # Baseline for comparison
│   ├── pipeline.py                         # NaiveRAGPipeline (build + answer)
│   ├── chunker.py                          # Fixed-size paragraph chunker
│   ├── embedder.py                         # MiniLM + FAISS IndexFlatIP
│   ├── retriever.py                        # Single-stage FAISS cosine search
│   └── generator.py                        # Direct Groq call, no abstention
├── frontend/
│   ├── templates/index.html                # Web UI
│   └── static/                             # CSS / JS
├── generate_dataset.py                     # Synthetic 5,648-ticket dataset generator
├── customer_support_5k.csv                 # Generated dataset (0% duplicates)
├── model_training2.py                      # Kaggle training notebook
├── requirements.txt
└── README.md
```

---

## Dependencies

```
flask
groq
faiss-cpu
sentence-transformers
setfit
scikit-learn==1.6.1
rank-bm25
spacy
pydantic
pandas
numpy
scipy
joblib
```

Install spaCy model separately:
```bash
python -m spacy download en_core_web_sm
```
