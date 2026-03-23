# ── INSTALL ALL REQUIRED PACKAGES ──────────────────────────────────────────
!pip install -q setfit sentence-transformers faiss-cpu rank_bm25 pydantic
!python -m spacy download en_core_web_sm -q
!pip install --upgrade setfit transformers datasets

# ── CRITICAL FIX: Monkey patch for transformers 5.0+ compatibility ─────────
import transformers.training_args
import socket
from datetime import datetime, timezone
import os as _os

def _default_logdir() -> str:
    current_time = datetime.now(timezone.utc).strftime("%b%d_%H-%M-%S")
    return _os.path.join("runs", current_time + "_" + socket.gethostname())

if not hasattr(transformers.training_args, 'default_logdir'):
    transformers.training_args.default_logdir = _default_logdir
    print("✅ Applied transformers 5.0 compatibility patch")

# ── Now proceed with imports ───────────────────────────────────────────────
import os, re, html, json, pickle, math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report

from setfit import SetFitModel, SetFitTrainer, TrainingArguments
from datasets import Dataset
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import spacy

print('✅ All packages imported successfully!')
# ── TEMPERATURE SCALING CLASS ─────────────────────────────────────────────
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

class TemperatureScaling:
    """Temperature scaling for model calibration"""
    
    def __init__(self):
        self.temperature = 1.0
        self.classes = None
    
    def _scale_logits(self, logits, temperature):
        """Apply temperature scaling to logits"""
        return logits / temperature
    
    def _softmax(self, logits):
        """Convert logits to probabilities"""
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def _nll_loss(self, temperature, logits, labels):
        """Negative log-likelihood loss for optimization"""
        scaled_logits = self._scale_logits(logits, temperature)
        probs = self._softmax(scaled_logits)
        
        # One-hot encode labels
        n_classes = logits.shape[1]
        labels_onehot = np.zeros((len(labels), n_classes))
        for i, label in enumerate(labels):
            label_idx = list(self.classes).index(label)
            labels_onehot[i, label_idx] = 1
        
        # Calculate NLL
        eps = 1e-10
        nll = -np.sum(labels_onehot * np.log(probs + eps)) / len(labels)
        return nll
    
    def fit(self, model, val_texts, val_labels):
        """Fit temperature parameter on validation set"""
        print('  Fitting temperature scaling...')

        self.classes = model.labels

        # Get logits from model (before softmax)
        if (model.model_head is not None
                and hasattr(model.model_head, 'decision_function')):
            embeddings = model.model_body.encode(val_texts)
            logits = model.model_head.decision_function(embeddings)
        else:
            probs = model.predict_proba(val_texts)
            if hasattr(probs, 'detach'):
                probs = probs.detach().cpu().numpy()
            else:
                probs = np.array(probs)
            eps = 1e-10
            logits = np.log(probs + eps)
        
        # Optimize temperature
        result = minimize_scalar(
            self._nll_loss,
            bounds=(0.1, 10.0),
            args=(logits, val_labels),
            method='bounded'
        )
        
        self.temperature = result.x
        print(f'  ✅ Optimal temperature: {self.temperature:.4f}')
        
        return self
    
    def transform(self, probs):
        """Apply temperature scaling to probabilities"""
        eps = 1e-10
        logits = np.log(probs + eps)
        scaled_logits = self._scale_logits(logits, self.temperature)
        return self._softmax(scaled_logits)
    
    def save(self, path):
        """Save temperature parameter"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'temperature': self.temperature,
                'classes': self.classes
            }, f)
        print(f'  ✅ Temperature saved to {path}')
    
    @classmethod
    def load(cls, path):
        """Load temperature parameter"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls()
        instance.temperature = data['temperature']
        instance.classes = data['classes']
        print(f'  ✅ Temperature loaded: {instance.temperature:.4f}')
        return instance

print('✅ Temperature Scaling class loaded!')
# ── PII SCRUBBER MODULE ───────────────────────────────────────────────────
REPLACEMENTS: Dict[str, str] = {
    "EMAIL": "[REDACTED_EMAIL]",
    "PHONE": "[REDACTED_PHONE]",
    "SSN": "[REDACTED_SSN]",
    "ZIP": "[REDACTED_ZIP]",
    "IP": "[REDACTED_IP]",
    "CREDIT_CARD": "[REDACTED_CC]",
    "PERSON": "[REDACTED_PERSON]",
    "ORG": "[REDACTED_ORG]",
    "GPE": "[REDACTED_LOCATION]",
    "LOC": "[REDACTED_LOCATION]",
}

NER_LABELS_TO_REDACT = {"PERSON", "ORG", "GPE", "LOC"}

PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("ZIP", re.compile(r"\b\d{5}(?:-\d{4})?\b")),
    ("IP", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
]

@dataclass
class ScrubStats:
    regex_matches: int = 0
    ner_matches: int = 0
    cells_changed: int = 0

class PIIScrubber:
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.nlp = spacy.load(model_name)

    def _apply_regex_scrub(self, text: str, stats: ScrubStats) -> str:
        scrubbed = text
        for label, pattern in PATTERNS:
            replacement = REPLACEMENTS[label]
            def _replacer(match):
                stats.regex_matches += 1
                return replacement
            scrubbed = pattern.sub(_replacer, scrubbed)
        return scrubbed

    def _apply_ner_scrub(self, text: str, stats: ScrubStats) -> str:
        doc = self.nlp(text)
        ents = [ent for ent in doc.ents if ent.label_ in NER_LABELS_TO_REDACT]
        if not ents:
            return text
        redacted = text
        for ent in sorted(ents, key=lambda e: e.start_char, reverse=True):
            replacement = REPLACEMENTS.get(ent.label_, "[REDACTED]")
            redacted = redacted[: ent.start_char] + replacement + redacted[ent.end_char :]
            stats.ner_matches += 1
        return redacted

    def scrub_text(self, text: str, stats: ScrubStats) -> str:
        regex_first = self._apply_regex_scrub(text, stats)
        return self._apply_ner_scrub(regex_first, stats)

    def normalize(self, text: str) -> str:
        text = html.unescape(text)
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        doc = self.nlp(text)
        text = " ".join(t.lemma_ for t in doc if not t.is_space)
        return text.strip()

    def scrub_dataframe(self, df: pd.DataFrame, text_columns: Iterable[str], 
                       structured_columns: Dict[str, str]) -> Tuple[pd.DataFrame, ScrubStats]:
        stats = ScrubStats()
        scrubbed_df = df.copy()

        for col, replacement in structured_columns.items():
            if col not in scrubbed_df.columns:
                continue
            original = scrubbed_df[col].astype("string")
            masked = original.where(original.isna(), replacement)
            changes = (original != masked).fillna(False).sum()
            stats.cells_changed += int(changes)
            scrubbed_df[col] = masked

        for col in text_columns:
            if col not in scrubbed_df.columns:
                continue
            def _scrub_cell(value):
                if pd.isna(value):
                    return value
                as_text = str(value)
                scrubbed = self.scrub_text(as_text, stats)
                normalized = self.normalize(scrubbed)
                if normalized != as_text:
                    stats.cells_changed += 1
                return normalized
            scrubbed_df[col] = scrubbed_df[col].apply(_scrub_cell)

        return scrubbed_df, stats

def age_to_bucket(value, bucket_size: int = 5):
    if pd.isna(value):
        return value
    try:
        age = int(float(value))
    except (ValueError, TypeError):
        return value
    lower = math.floor(age / bucket_size) * bucket_size
    upper = lower + bucket_size
    return f"{lower}-{upper}"

print('✅ PII Scrubber module loaded!')
# ── CONTROL FLAGS ──────────────────────────────────────────────────────────
SKIP_BASELINE = False        # ✅ Load from saved dataset
SKIP_SETFIT = False          # ✅ Load from saved dataset
SKIP_CALIBRATION = False     # ✅ Load from saved dataset
ONLY_REBUILD_KB = False
LOAD_FROM_DATASET = False    # ✅ MUST BE TRUE
USE_PII_SCRUBBER = False
# ───────────────────────────────────────────────────────────────────────────

if ONLY_REBUILD_KB:
    SKIP_BASELINE = True
    SKIP_SETFIT = True
    SKIP_CALIBRATION = True
    LOAD_FROM_DATASET = True

print(f'🎯 Configuration:')
print(f'  SKIP_BASELINE: {SKIP_BASELINE}')
print(f'  SKIP_SETFIT: {SKIP_SETFIT}')
print(f'  SKIP_CALIBRATION: {SKIP_CALIBRATION}')
print(f'  ONLY_REBUILD_KB: {ONLY_REBUILD_KB}')
print(f'  LOAD_FROM_DATASET: {LOAD_FROM_DATASET}')
print(f'  USE_PII_SCRUBBER: {USE_PII_SCRUBBER}')
# ── PATHS WITH DATASET LOADING SUPPORT ────────────────────────────────────
INPUT_DIR = Path('/kaggle/input/datasets/chantbappu/customer-support-ticket-v3')
MODELS_INPUT = Path('/kaggle/input/datasets/chantbappu/trained-for-decompute-v1')  # ✅ CORRECT PATH
KB_INPUT = Path('/kaggle/input/datasets/saadmakhdoom/ecommerce-faq-chatbot-dataset')
OUTPUT_DIR = Path('/kaggle/working')
MODELS_DIR = OUTPUT_DIR / 'models'
DATA_DIR = OUTPUT_DIR / 'processed'

# Create directories
for d in [MODELS_DIR, DATA_DIR, 
          MODELS_DIR / 'setfit_category', 
          MODELS_DIR / 'setfit_priority']:
    d.mkdir(parents=True, exist_ok=True)

# Load pre-trained models if requested
if LOAD_FROM_DATASET and MODELS_INPUT and MODELS_INPUT.exists():
    import shutil
    print('📦 Loading pre-trained models from dataset...')
    print(f'   Source: {MODELS_INPUT}')
    print(f'   Destination: {MODELS_DIR}')
    
    copied_count = 0
    for item in MODELS_INPUT.rglob('*'):
        if item.is_file():
            dest = MODELS_DIR / item.relative_to(MODELS_INPUT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)
            copied_count += 1
    
    print(f'✅ Models loaded successfully! ({copied_count} files)')
    
    # Verify what was loaded
    print('\n📁 Loaded files:')
    for p in sorted(MODELS_DIR.rglob('*')):
        if p.is_file():
            print(f'  {p.relative_to(OUTPUT_DIR)}')
elif LOAD_FROM_DATASET:
    print('⚠️ LOAD_FROM_DATASET=True but dataset not found!')
    print(f'   Expected path: {MODELS_INPUT}')
    print('   Will train from scratch...')
    LOAD_FROM_DATASET = False
else:
    print('ℹ️ LOAD_FROM_DATASET=False - will train from scratch')

print(f'\n📁 Output directory: {OUTPUT_DIR}')
print(f'📁 Models directory exists: {MODELS_DIR.exists()}')

# ── FIX NESTED DIRECTORY STRUCTURE ──────────────────────────────────────
import shutil
from pathlib import Path

MODELS_DIR = Path('/kaggle/working/models')
nested_models_dir = MODELS_DIR / 'models'

if nested_models_dir.exists():
    print("🔧 Fixing nested directory structure...")
    
    for item in nested_models_dir.iterdir():
        dest = MODELS_DIR / item.name
        
        if item.is_file():
            if not dest.exists():
                shutil.move(str(item), str(MODELS_DIR))
                print(f"  ✅ Moved {item.name} to root")
            else:
                print(f"  ⚠️  Skipping {item.name} (already exists)")
        
        elif item.is_dir():
            if dest.exists():
                # Merge directory contents
                for sub_item in item.rglob('*'):
                    if sub_item.is_file():
                        sub_dest = dest / sub_item.relative_to(item)
                        sub_dest.parent.mkdir(parents=True, exist_ok=True)
                        if not sub_dest.exists():
                            shutil.copy2(str(sub_item), str(sub_dest))
                print(f"  ✅ Merged {item.name}/ contents")
            else:
                shutil.move(str(item), str(MODELS_DIR))
                print(f"  ✅ Moved {item.name}/ to root")
    
    # Force remove nested folder
    shutil.rmtree(nested_models_dir)
    print("✅ Directory structure fixed!")
else:
    print("ℹ️  Directory structure already correct")

from pathlib import Path
MODELS_DIR = Path('/kaggle/working/models')

print('Files in /kaggle/working/models/:')
if MODELS_DIR.exists():
    for f in sorted(MODELS_DIR.rglob('*')):
        if f.is_file():
            print(f'  {f.relative_to(Path("/kaggle/working"))}')
else:
    print('  ❌ Directory does not exist!')

# ── Config ─────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
SETFIT_SAMPLES = 30  # num_iterations: contrastive pairs per class
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# ── GPU Check ──────────────────────────────────────────────────────────────
import torch
if torch.cuda.is_available():
    print(f'🟢 GPU available: {torch.cuda.get_device_name(0)}')
    print(f'   GPU count: {torch.cuda.device_count()}')
    print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'   Training will use: cuda:0')
else:
    print('🔴 NO GPU detected — training will run on CPU (will be slow!)')

# Load spaCy models
nlp_lemma = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp_ner = spacy.load('en_core_web_sm')

print("✅ Setup complete!")
print(f"SetFit version: {__import__('setfit').__version__}")
print(f"Sentence Transformers version: {__import__('sentence_transformers').__version__}")

# ── LOAD DATA WITH PII SCRUBBING ──────────────────────────────────────────
print('🔒 Loading and scrubbing PII from customer support tickets...')

# Load raw data
train_raw = pd.read_csv(INPUT_DIR / 'customer_support_5k.csv')

# Rename columns to match expected format
# Note: Ticket Subject was dropped from v2 (it leaked the category label)
train_raw = train_raw.rename(columns={
    'Ticket Type': 'category',
    'Ticket Priority': 'priority',
    'Ticket Description': 'message',
})

print(f'Raw train shape: {train_raw.shape}')

# Age bucketing (if column exists)
if 'Customer Age' in train_raw.columns:
    train_raw['Customer Age'] = train_raw['Customer Age'].apply(age_to_bucket)
    print('✅ Age bucketing applied')

# Apply PII scrubbing
if USE_PII_SCRUBBER:
    scrubber = PIIScrubber(model_name='en_core_web_sm')
    
    structured_columns = {
        'Customer Name': '[REDACTED_PERSON]',
        'Customer Email': '[REDACTED_EMAIL]',
    }
    
    text_columns = ['message']  # subject excluded — leaks category label
    if 'Resolution' in train_raw.columns:
        text_columns.append('Resolution')
    
    scrubbed_df, scrub_stats = scrubber.scrub_dataframe(
        df=train_raw,
        text_columns=text_columns,
        structured_columns=structured_columns,
    )
    
    print(f'\n🛡️ PII Scrubbing Summary:')
    print(f'  Regex matches: {scrub_stats.regex_matches}')
    print(f'  NER matches: {scrub_stats.ner_matches}')
    print(f'  Cells changed: {scrub_stats.cells_changed}')
    
    train_raw = scrubbed_df
else:
    # Simple preprocessing without PII scrubbing
    # NOTE: subject is intentionally excluded — it is a human-written proxy for the
    # category label (e.g. "Invoice discrepancy" → Billing inquiry) and causes
    # perfect-score data leakage if included. Only description is used for training.
    def preprocess(row):
        message = row['message'] if pd.notna(row['message']) else ""
        text = message.strip()
        if not text:
            return ""
        text = html.unescape(text).lower()
        text = re.sub(r"[^\w\s]", " ", text)
        doc = nlp_lemma(text)
        return " ".join([token.lemma_ for token in doc if not token.is_space]).strip()
    
    train_raw['cleaned'] = train_raw.apply(preprocess, axis=1)

# Create cleaned column if not exists (description only — subject excluded)
if 'cleaned' not in train_raw.columns:
    train_raw['cleaned'] = train_raw['message'].fillna('').str.strip()

print(f'\n✅ Preprocessing complete. Sample cleaned text:')
print(f'  {train_raw["cleaned"].iloc[0][:200]}...')

print(f'\n📊 Category distribution:')
print(train_raw['category'].value_counts())
print(f'\n📊 Priority distribution:')
print(train_raw['priority'].value_counts())

# ── 60/20/20 stratified split ──────────────────────────────────────────────
train_df, temp_df = train_test_split(
    train_raw, test_size=0.40, random_state=RANDOM_SEED, 
    stratify=train_raw['category']
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=RANDOM_SEED, 
    stratify=temp_df['category']
)

print(f'Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}')

# Verify stratification
for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
    print(f'\n{name} category %:')
    print(df['category'].value_counts(normalize=True).round(3))

# Prepare features and labels
X_train = train_df['cleaned'].values
X_val = val_df['cleaned'].values
X_test = test_df['cleaned'].values

y_cat_train = train_df['category'].values
y_cat_val = val_df['category'].values
y_cat_test = test_df['category'].values

y_pri_train = train_df['priority'].values
y_pri_val = val_df['priority'].values
y_pri_test = test_df['priority'].values

# Check what's in models folder
from pathlib import Path
MODELS_DIR = Path('/kaggle/working/models')

print('Files in /kaggle/working/models/:')
if MODELS_DIR.exists():
    for f in MODELS_DIR.rglob('*'):
        if f.is_file():
            print(f'  {f.relative_to(Path("/kaggle/working"))}')
else:
    print('  ❌ Directory does not exist!')

# ── TF-IDF + LogReg baseline WITH CHAR N-GRAMS ───────────────────────────
if not SKIP_BASELINE:
    print('🔥 Training baseline models with word + char n-grams...')
    
    from sklearn.pipeline import FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Word-level n-grams
    word_vectorizer = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 2), max_features=20000,
        sublinear_tf=True, min_df=2, max_df=0.95
    )
    
    # Character-level n-grams (NEW for typo/abbreviation handling)
    char_vectorizer = TfidfVectorizer(
        analyzer='char', ngram_range=(3, 5), max_features=10000,
        sublinear_tf=True, min_df=2
    )
    
    # Combine
    combined_vectorizer = FeatureUnion([
        ('word_ngrams', word_vectorizer),
        ('char_ngrams', char_vectorizer)
    ])
    
    # Category pipeline
    cat_pipeline = Pipeline([
        ('tfidf', combined_vectorizer),
        ('clf', LogisticRegression(max_iter=1000, C=5.0, class_weight='balanced', random_state=RANDOM_SEED))
    ])
    cat_pipeline.fit(X_train, y_cat_train)

    # Priority pipeline (reuse vectorizer config)
    pri_pipeline = Pipeline([
        ('tfidf', combined_vectorizer),
        ('clf', LogisticRegression(max_iter=1000, C=5.0, class_weight='balanced', random_state=RANDOM_SEED))
    ])
    pri_pipeline.fit(X_train, y_pri_train)

    print('=== Baseline TF-IDF + LogReg (word + char n-grams) ===')
    print('Category:')
    print(classification_report(y_cat_test, cat_pipeline.predict(X_test)))
    print('Priority:')
    print(classification_report(y_pri_test, pri_pipeline.predict(X_test)))

    with open(MODELS_DIR / 'baseline_cat_pipeline.pkl', 'wb') as f:
        pickle.dump(cat_pipeline, f)
    with open(MODELS_DIR / 'baseline_pri_pipeline.pkl', 'wb') as f:
        pickle.dump(pri_pipeline, f)
    print('✅ Baseline models saved')
    
else:
    print('⏩ Skipping baseline training...')
    
    # Try both paths (in case directory fix wasn't run)
    baseline_cat_path = MODELS_DIR / 'baseline_cat_pipeline.pkl'
    baseline_pri_path = MODELS_DIR / 'baseline_pri_pipeline.pkl'
    
    # Fallback to nested location
    if not baseline_cat_path.exists():
        baseline_cat_path = MODELS_DIR / 'models' / 'baseline_cat_pipeline.pkl'
    if not baseline_pri_path.exists():
        baseline_pri_path = MODELS_DIR / 'models' / 'baseline_pri_pipeline.pkl'
    
    with open(baseline_cat_path, 'rb') as f:
        cat_pipeline = pickle.load(f)
    with open(baseline_pri_path, 'rb') as f:
        pri_pipeline = pickle.load(f)
    print('✅ Baseline models loaded from disk')

# ── SetFit Training Function ───────────────────────────────────────────────
def train_setfit(train_df, val_df, test_df, label_col, save_path,
                 num_iterations=20, num_epochs=1,
                 model_name='sentence-transformers/all-MiniLM-L6-v2',
                 batch_size=16):
    import os

    save_path_str = str(save_path)
    
    # Resume logic - check if model already exists
    if os.path.exists(save_path_str) and os.path.exists(os.path.join(save_path_str, 'config_sentence_transformers.json')):
        print(f"⏩ Found fully trained {label_col} model! Loading...")
        model = SetFitModel.from_pretrained(save_path_str)
        return None, model, None
    
    # Check for checkpoints
    if os.path.exists(save_path_str):
        checkpoints = [d for d in os.listdir(save_path_str) if d.startswith("checkpoint")]
        if checkpoints:
            latest_checkpoint = os.path.join(save_path_str, sorted(checkpoints)[-1])
            print(f"🔄 Found checkpoint at {latest_checkpoint}! Loading...")
            model = SetFitModel.from_pretrained(latest_checkpoint)
            model.save_pretrained(save_path_str)
            return None, model, None
    
    # Get labels
    labels = sorted(train_df[label_col].unique().tolist())

    print(f"Labels for {label_col}: {labels}")

    # Prepare datasets
    train_dataset = Dataset.from_pandas(
        train_df[[label_col, 'cleaned']].rename(columns={'cleaned': 'text', label_col: 'label'})
    )
    val_dataset = Dataset.from_pandas(
        val_df[[label_col, 'cleaned']].rename(columns={'cleaned': 'text', label_col: 'label'})
    )

    # Load model - from_pretrained only works for SetFit repos, not plain sentence transformers
    print(f"Loading model: {model_name}")
    import torch
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    try:
        model = SetFitModel.from_pretrained(model_name, labels=labels)
        print("  Loaded as SetFit model")
    except Exception:
        # Plain sentence transformer repo: build SetFit model manually
        model_body = SentenceTransformer(model_name, device=device)
        model = SetFitModel(model_body=model_body, labels=labels)
        print("  Built SetFit model from SentenceTransformer")
    # Pin to cuda:0 to avoid DataParallel StopIteration bug with 2x GPU
    model = model.to(device)
    print(f"  Model device: {next(model.model_body.parameters()).device}")

    # Ensure model_head is initialized before training (required by setfit >= 1.0)
    if model.model_head is None:
        from sklearn.linear_model import LogisticRegression
        model.model_head = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        print("  ✅ Initialized LogReg model_head before training")

    # Create trainer (compatible with both old and new SetFit API)
    print(f"Training {label_col} model...")
    try:
        # SetFit 1.0+ API
        training_args = TrainingArguments(
            num_epochs=num_epochs,
            num_iterations=num_iterations,
            batch_size=batch_size,
            seed=RANDOM_SEED,
            body_learning_rate=2e-5,
            head_learning_rate=1e-2,
        )
        trainer = SetFitTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        print("  Using SetFit 1.0+ API (TrainingArguments)")
    except TypeError:
        # Older SetFit API
        trainer = SetFitTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            batch_size=batch_size,
            num_iterations=num_iterations,
            num_epochs=num_epochs,
            seed=RANDOM_SEED,
        )
        print("  Using legacy SetFit API (direct kwargs)")
    
    # Train
    import warnings
    warnings.filterwarnings('ignore')
    trainer.train()

    # Verify model_head was fitted (needed for calibration + tier_2 deployment)
    if model.model_head is None:
        print(f"  ⚠️ model_head is None after training — fitting LogReg head manually...")
        from sklearn.linear_model import LogisticRegression
        embeddings = model.model_body.encode(train_dataset['text'])
        head = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        head.fit(embeddings, train_dataset['label'])
        model.model_head = head
        print(f"  ✅ LogReg head fitted with {len(train_dataset)} samples")
    else:
        print(f"  ✅ model_head type: {type(model.model_head).__name__}")

    # Evaluate
    metrics = trainer.evaluate()

    # Save model
    print(f"Saving {label_col} model...")
    model.save_pretrained(save_path_str)

    # Verify save actually wrote files
    saved_files = list(Path(save_path_str).rglob('*'))
    saved_size = sum(f.stat().st_size for f in saved_files if f.is_file())
    print(f"✅ {label_col} model saved to {save_path_str} ({saved_size / 1024 / 1024:.1f} MB, {len(saved_files)} files)")
    
    return trainer, model, metrics


# ── Execute SetFit Training ───────────────────────────────────────────────
if not SKIP_SETFIT:
    print("Checking Category Model...")
    setfit_cat_trainer, setfit_cat_model, cat_metrics = train_setfit(
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_col='category',
        save_path=MODELS_DIR / 'setfit_category',
        num_iterations=SETFIT_SAMPLES, num_epochs=3,
        model_name=EMBED_MODEL, batch_size=16
    )

    print("\nChecking Priority Model...")
    setfit_pri_trainer, setfit_pri_model, pri_metrics = train_setfit(
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_col='priority',
        save_path=MODELS_DIR / 'setfit_priority',
        num_iterations=SETFIT_SAMPLES, num_epochs=3,
        model_name=EMBED_MODEL, batch_size=16
    )
    print('✅ SetFit models trained and saved')
    
else:
    print('⏩ Skipping SetFit training...')
    
    # Try both paths
    setfit_cat_path = MODELS_DIR / 'setfit_category'
    setfit_pri_path = MODELS_DIR / 'setfit_priority'
    
    # Fallback to nested location
    if not setfit_cat_path.exists():
        setfit_cat_path = MODELS_DIR / 'models' / 'setfit_category'
    if not setfit_pri_path.exists():
        setfit_pri_path = MODELS_DIR / 'models' / 'setfit_priority'
    
    setfit_cat_model = SetFitModel.from_pretrained(str(setfit_cat_path))
    setfit_pri_model = SetFitModel.from_pretrained(str(setfit_pri_path))
    print('✅ SetFit models loaded from disk')

# ── LOAD CALIBRATORS (WHEN SKIP_CALIBRATION=True) ─────────────────────────
if SKIP_CALIBRATION:
    print('📦 Loading saved calibrators...')
    
    cat_calibrator_path = MODELS_DIR / 'cat_calibrators.pkl'
    pri_calibrator_path = MODELS_DIR / 'pri_calibrators.pkl'
    
    # Check nested structure (from dataset copy issue)
    if not cat_calibrator_path.exists():
        cat_calibrator_path = MODELS_DIR / 'models' / 'cat_calibrators.pkl'
    if not pri_calibrator_path.exists():
        pri_calibrator_path = MODELS_DIR / 'models' / 'pri_calibrators.pkl'
    
    if cat_calibrator_path.exists() and pri_calibrator_path.exists():
        with open(cat_calibrator_path, 'rb') as f:
            cat_calibrators = pickle.load(f)
        with open(pri_calibrator_path, 'rb') as f:
            pri_calibrators = pickle.load(f)
        print('✅ Calibrators loaded from disk')
    else:
        print('⚠️  Calibrator files not found! Setting SKIP_CALIBRATION=False to rebuild...')
        print(f'   Expected: {cat_calibrator_path}')
        print(f'   Expected: {pri_calibrator_path}')
        SKIP_CALIBRATION = False
else:
    print('ℹ️  SKIP_CALIBRATION=False - will train calibrators')


from pathlib import Path
MODELS_DIR = Path('/kaggle/working/models')
print('🔍 Searching for calibrator files...')
for f in MODELS_DIR.rglob('*calib*'):
    print(f'  {f.relative_to(Path("/kaggle/working"))}')
for f in MODELS_DIR.rglob('*.pkl'):
    if 'calib' in f.name.lower() or 'temp' in f.name.lower():
        print(f'  {f.relative_to(Path("/kaggle/working"))}')


# ── TWO-STAGE CALIBRATION (Temperature + Isotonic) ────────────────────────
def fit_calibrators(model, val_texts, val_labels, save_path, 
                   use_temperature=True, use_isotonic=True):
    """
    Two-stage calibration:
    1. Temperature scaling (global calibration)
    2. Isotonic regression (per-class calibration)
    """
    from sklearn.isotonic import IsotonicRegression
    import pickle
    
    classes = model.labels
    print(f'🔧 Fitting calibrators for {len(classes)} classes...')
    
    # ── STAGE 1: Temperature Scaling ──────────────────────────────────────
    if use_temperature:
        temp_scaler = TemperatureScaling()
        temp_scaler.fit(model, val_texts, val_labels)
        temp_scaler.save(save_path.parent / 'temperature_scaling.pkl')
    else:
        temp_scaler = None
    
    # ── STAGE 2: Isotonic Regression ──────────────────────────────────────
    if use_isotonic:
        if temp_scaler and model.model_head is not None and hasattr(model.model_head, 'decision_function'):
            val_embeddings = model.model_body.encode(val_texts)
            raw_logits = model.model_head.decision_function(val_embeddings)
            scaled_logits = raw_logits / temp_scaler.temperature
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            val_proba = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        else:
            val_proba = model.predict_proba(val_texts)
            if hasattr(val_proba, 'detach'):
                val_proba = val_proba.detach().cpu().numpy()
            else:
                val_proba = np.array(val_proba)
        
        calibrators = {}
        for i, cls in enumerate(classes):
            binary_labels = (np.array(val_labels) == cls).astype(int)
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(val_proba[:, i], binary_labels)
            calibrators[cls] = iso
        
        print(f'  ✅ Isotonic calibrators fitted for {len(classes)} classes')
    else:
        calibrators = None
    
    # ── Save Both Calibrators ─────────────────────────────────────────────
    # Include 'head' key so tier_2.py _load_bundle() can resolve the LogReg head
    head = model.model_head if model.model_head is not None else None
    with open(save_path, 'wb') as f:
        pickle.dump({
            'head': head,
            'calibrators': calibrators,
            'classes': classes,
            'temperature': temp_scaler.temperature if temp_scaler else None,
            'use_temperature': use_temperature,
            'use_isotonic': use_isotonic
        }, f)
    print(f'✅ Calibrators saved to {save_path}')

    return calibrators, temp_scaler if use_temperature else None

# ── Execute Two-Stage Calibration ────────────────────────────────────────
cat_calib_path = MODELS_DIR / 'calibration_category.pkl'
pri_calib_path = MODELS_DIR / 'calibration_priority.pkl'

# Category calibration
print('🔧 Category Calibration:')
if not cat_calib_path.exists() or not SKIP_CALIBRATION:
    cat_calibrators, cat_temp_scaler = fit_calibrators(
        model=setfit_cat_model,
        val_texts=val_df['cleaned'].tolist(),
        val_labels=val_df['category'].tolist(),
        save_path=cat_calib_path,
        use_temperature=True,
        use_isotonic=True
    )
else:
    print('⏩ Loading category calibrators from disk...')
    with open(cat_calib_path, 'rb') as f:
        cat_data = pickle.load(f)
    cat_calibrators = cat_data['calibrators']
    if cat_data.get('use_temperature', False):
        cat_temp_scaler = TemperatureScaling.load(
            cat_calib_path.parent / 'temperature_scaling.pkl'
        )
    else:
        cat_temp_scaler = None

# Priority calibration
print('\n🔧 Priority Calibration:')
if not pri_calib_path.exists() or not SKIP_CALIBRATION:
    pri_calibrators, pri_temp_scaler = fit_calibrators(
        model=setfit_pri_model,
        val_texts=val_df['cleaned'].tolist(),
        val_labels=val_df['priority'].tolist(),
        save_path=pri_calib_path,
        use_temperature=True,
        use_isotonic=True
    )
else:
    print('⏩ Loading priority calibrators from disk...')
    with open(pri_calib_path, 'rb') as f:
        pri_data = pickle.load(f)
    pri_calibrators = pri_data['calibrators']
    if pri_data.get('use_temperature', False):
        pri_temp_scaler = TemperatureScaling.load(
            pri_calib_path.parent / 'temperature_scaling.pkl'
        )
    else:
        pri_temp_scaler = None

print('\n✅ Two-stage calibration complete!')

# ── TWO-STAGE CALIBRATED PREDICTION ──────────────────────────────────────
def calibrated_predict(model, calibrators, temp_scaler, texts):
    """Apply two-stage calibration: Temperature scaling + Isotonic regression"""
    classes = model.labels
    raw_proba = model.predict_proba(texts)
    
    # ── THE FIX: Convert PyTorch Tensor to NumPy Array ──
    if hasattr(raw_proba, 'detach'):
        raw_proba = raw_proba.detach().cpu().numpy()
    elif hasattr(raw_proba, 'numpy'):
        raw_proba = raw_proba.numpy()
    else:
        raw_proba = np.array(raw_proba)
    # ────────────────────────────────────────────────────
    
    # STAGE 1: Temperature Scaling
    if temp_scaler is not None:
        temp_proba = temp_scaler.transform(raw_proba)
        print(f'  Applied temperature scaling (T={temp_scaler.temperature:.4f})')
    else:
        temp_proba = raw_proba
    
    # STAGE 2: Isotonic Regression
    if calibrators is not None:
        cal_proba = np.zeros_like(temp_proba)
        for i, cls in enumerate(classes):
            cal_proba[:, i] = calibrators[cls].predict(temp_proba[:, i])
        row_sums = cal_proba.sum(axis=1, keepdims=True)
        cal_proba = cal_proba / np.maximum(row_sums, 1e-9)
        print(f'  Applied isotonic regression for {len(classes)} classes')
    else:
        cal_proba = temp_proba
    
    pred_idx = np.argmax(cal_proba, axis=1)
    preds = [classes[i] for i in pred_idx]
    confs = cal_proba.max(axis=1)
    
    return preds, confs, cal_proba

# ── Evaluate on test set ──────────────────────────────────────────────────
print('\n=== Final test results — SetFit + Two-Stage Calibration ===')

cat_preds, cat_confs, cat_cal_proba = calibrated_predict(
    setfit_cat_model, cat_calibrators, cat_temp_scaler, 
    test_df['cleaned'].tolist()
)

pri_preds, pri_confs, pri_cal_proba = calibrated_predict(
    setfit_pri_model, pri_calibrators, pri_temp_scaler, 
    test_df['cleaned'].tolist()
)

print('\nCategory:')
print(classification_report(y_cat_test, cat_preds))
print('Priority:')
print(classification_report(y_pri_test, pri_preds))

# ── Calculate ECE (Expected Calibration Error) ───────────────────────────
def calculate_ece(y_true, y_pred_proba, labels, n_bins=10):
    """Calculate Expected Calibration Error"""
    confidences = np.max(y_pred_proba, axis=1)
    predictions = np.argmax(y_pred_proba, axis=1)
    
    accuracies = (predictions == np.array([list(labels).index(y) 
                                           for y in y_true])).astype(int)
    # ───────────────────────────────────────────────────────────────
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    
    return ece

# Now pass the correct labels for each model when calling the function!
cat_ece = calculate_ece(y_cat_test, cat_cal_proba, setfit_cat_model.labels)
pri_ece = calculate_ece(y_pri_test, pri_cal_proba, setfit_pri_model.labels)

print(f'\n📊 Calibration Metrics:')
print(f'  Category ECE: {cat_ece:.4f}')
print(f'  Priority ECE: {pri_ece:.4f}')
print(f'  (Lower ECE = better calibration)')

# ── Save manifest (FIXED: preserves kb_info + calibration info) ──────────
baseline_cat_preds = cat_pipeline.predict(X_test)
baseline_pri_preds = pri_pipeline.predict(X_test)

# ✅ LOAD existing manifest first (preserves kb_info from KB re-build cell)
manifest_path = OUTPUT_DIR / 'training_manifest.json'
if manifest_path.exists():
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    print('✅ Loaded existing manifest (preserves kb_info)')
else:
    manifest = {}
    print('⚠️ No existing manifest found, creating new one')

# Update with model metrics and artifacts
manifest.update({
    'splits': {'train': len(train_df), 'val': len(val_df), 'test': len(test_df)},
    'kb_info': {
        'source': 'Ecommerce_FAQ_Chatbot_dataset.json',
        'total_chunks': (
            len(all_chunks) if 'all_chunks' in dir()
            else sum(1 for _ in open(OUTPUT_DIR / 'processed/kb_chunks.jsonl')) if (OUTPUT_DIR / 'processed/kb_chunks.jsonl').exists()
            else None
        ),
        'path': str(KB_PATH) if 'KB_PATH' in dir() else None
    },
    'calibration_info': {
        'method': 'two_stage',
        'stage1': 'temperature_scaling',
        'stage2': 'isotonic_regression',
        'category_temperature': float(cat_temp_scaler.temperature) if cat_temp_scaler else None,
        'priority_temperature': float(pri_temp_scaler.temperature) if pri_temp_scaler else None,
        'category_ece': round(cat_ece, 4),
        'priority_ece': round(pri_ece, 4),
    },
    'artifacts': {
        'baseline_cat':        'models/baseline_cat_pipeline.pkl',
        'baseline_pri':        'models/baseline_pri_pipeline.pkl',
        'setfit_category':     'models/setfit_category/',
        'setfit_priority':     'models/setfit_priority/',
        'calibration_cat':     'models/calibration_category.pkl',
        'calibration_pri':     'models/calibration_priority.pkl',
        'temperature_cat':     'models/temperature_scaling.pkl',
        'temperature_pri':     'models/temperature_scaling.pkl',
        'tier1_category_head': 'models/baseline_cat_pipeline.pkl',
        'tier1_priority_head': 'models/baseline_pri_pipeline.pkl',
        'bm25_index':          'models/bm25_index.pkl',
        'faiss_index':         'models/faiss_index.bin',
        'chunk_embeddings':    'models/chunk_embeddings.npy',
        'kb_chunks':           'processed/kb_chunks.jsonl',
    },
    'thresholds': {
        'tier1_confidence': 0.90,
        'tier2_confidence': 0.45,
        'abstain_rerank':   0.35,
    },
    'metrics': {
        'setfit_category_macro_f1': round(f1_score(y_cat_test, cat_preds, average='macro'), 4),
        'setfit_priority_macro_f1': round(f1_score(y_pri_test, pri_preds, average='macro'), 4),
        'baseline_category_macro_f1': round(f1_score(y_cat_test, baseline_cat_preds, average='macro'), 4),
        'baseline_priority_macro_f1': round(f1_score(y_pri_test, baseline_pri_preds, average='macro'), 4),
    }
})

# Save updated manifest
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print('\n✅ Training complete! Files saved to /kaggle/working/')
print('\nModel Performance Summary:')
print(f"  SetFit Category F1:   {manifest['metrics']['setfit_category_macro_f1']}")
print(f"  SetFit Priority F1:   {manifest['metrics']['setfit_priority_macro_f1']}")
print(f"  Baseline Category F1: {manifest['metrics']['baseline_category_macro_f1']}")
print(f"  Baseline Priority F1: {manifest['metrics']['baseline_priority_macro_f1']}")

# Show KB info
print(f"\n📚 Knowledge Base Info:")
print(f"  Source: {manifest['kb_info']['source']}")
print(f"  Total Chunks: {manifest['kb_info']['total_chunks']}")

# Show Calibration info
print(f"\n🔧 Calibration Info:")
print(f"  Method: {manifest['calibration_info']['method']}")
print(f"  Category Temperature: {manifest['calibration_info']['category_temperature']}")
print(f"  Priority Temperature: {manifest['calibration_info']['priority_temperature']}")
print(f"  Category ECE: {manifest['calibration_info']['category_ece']}")
print(f"  Priority ECE: {manifest['calibration_info']['priority_ece']}")

print('\nFiles in /kaggle/working/models/:')
for p in sorted(MODELS_DIR.rglob('*')):
    if p.is_file():
        print(f'  {p.relative_to(OUTPUT_DIR)}')


# ── Re-save Tier 1 as proper LabelHead (with fitted rules) ────────────────
# Requires backend source uploaded as a Kaggle dataset (flat: Rules.py + tier_1.py).
BACKEND_DATASET = Path('/kaggle/input/datasets/chantbappu/ticket-classifier-backend')

# Create package structure from flat files so imports work
if BACKEND_DATASET.exists() and (BACKEND_DATASET / 'tier_1.py').exists():
    import sys, joblib, shutil
    _tmp_src = Path('/kaggle/working/_backend_src')
    (_tmp_src / 'Classification').mkdir(parents=True, exist_ok=True)
    (_tmp_src / 'Scrubber').mkdir(parents=True, exist_ok=True)
    shutil.copy2(BACKEND_DATASET / 'tier_1.py', _tmp_src / 'Classification' / 'tier_1.py')
    shutil.copy2(BACKEND_DATASET / 'Rules.py', _tmp_src / 'Scrubber' / 'Rules.py')
    if str(_tmp_src) not in sys.path:
        sys.path.insert(0, str(_tmp_src))
    print('✅ Created package structure from flat dataset files')

    from Classification.tier_1 import LabelHead, DataDrivenRules, CalibratedLogReg

    def wrap_as_label_head(sklearn_pipeline, X_train, y_train, threshold=0.90):
        """Wrap an already-trained sklearn Pipeline into a LabelHead with fitted rules."""
        head = LabelHead(threshold=threshold)
        normalized = [t.lower().strip() for t in X_train]
        head.rules.fit(normalized, list(y_train))
        head.logreg = CalibratedLogReg.__new__(CalibratedLogReg)
        head.logreg.model = sklearn_pipeline
        head.logreg.classes_ = np.asarray(sklearn_pipeline.classes_)
        head._is_fitted = True
        return head

    cat_head = wrap_as_label_head(cat_pipeline, X_train, y_cat_train)
    pri_head = wrap_as_label_head(pri_pipeline, X_train, y_pri_train)

    joblib.dump(cat_head, MODELS_DIR / 'baseline_cat_pipeline.pkl')
    joblib.dump(pri_head, MODELS_DIR / 'baseline_pri_pipeline.pkl')

    print(f'✅ Tier 1 LabelHead models saved with fitted keyword rules')
    for name, head in [('category', cat_head), ('priority', pri_head)]:
        print(f'  {name}: {len(head.rules.class_keywords)} classes, '
              f'keywords: {list(head.rules.class_keywords.keys())}')
else:
    print('ℹ️  Backend source not found on Kaggle — skipping LabelHead wrapping.')
    print('    Tier 1 pkl files are raw sklearn Pipelines (auto-wrapped at load time).')
    print('    To include fitted keyword rules, upload Rules.py + tier_1.py as a Kaggle dataset.')

print('\n🎉 Ready for deployment!')