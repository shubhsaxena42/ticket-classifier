from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from Scrubber.Rules import TicketPrediction


# Train/serve preprocessing contract for pipeline parity.
# 1) Scrubber performs PII redaction + normalization + lemmatization.
# 2) Tier1 normalization applies lightweight cleanup and placeholder stripping.
PREPROCESSING_ORDER: Tuple[str, ...] = ("scrubber", "tier1_normalize")


def normalize_ticket_text(text: str) -> str:
	"""Tier-1 text normalization; lemmatization is expected from scrub node."""
	normalized = str(text).lower().strip()
	# Remove synthetic placeholders frequently present in this dataset.
	normalized = re.sub(r"\{[^{}]+\}", " ", normalized)
	normalized = re.sub(r"\s+", " ", normalized)
	return normalized.strip()


def compose_ticket_text(df: pd.DataFrame) -> pd.Series:
	"""Build model text from stable ticket fields used at inference time.

	Subject is excluded: it is a human-written proxy for the category label and
	causes data leakage. Resolution is excluded because it is missing for open
	tickets and populated only after closure.
	"""
	# Subject excluded: model trained on description only (subject leaks category label)
	description = df.get("Ticket Description", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
	return description.str.strip().str.lower().apply(normalize_ticket_text)


@dataclass
class LabelPrediction:
	predicted_label: str
	confidence: float
	tier_used: Literal["rules", "logreg"]
	routing_action: Literal["auto_route", "escalate_to_tier2"]
	abstain_flag: bool


class DataDrivenRules:
	"""Extract top discriminative n-grams per class and apply regex boundary matching."""

	def __init__(self, top_k: int = 10, ngram_range: Tuple[int, int] = (1, 2)) -> None:
		self.top_k = top_k
		self.vectorizer = TfidfVectorizer(ngram_range=ngram_range)
		self.class_keywords: Dict[str, List[str]] = {}
		self._class_patterns: Dict[str, List[re.Pattern[str]]] = {}

	def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "DataDrivenRules":
		if len(texts) == 0:
			raise ValueError("Cannot fit rules on an empty dataset.")

		x = self.vectorizer.fit_transform(texts)
		y = np.asarray(labels)
		classes = np.unique(y)
		feature_names = np.asarray(self.vectorizer.get_feature_names_out())

		class_means: Dict[str, np.ndarray] = {}
		for cls in classes:
			mask = y == cls
			class_means[cls] = np.asarray(x[mask].mean(axis=0)).ravel()

		for cls in classes:
			this_mean = class_means[cls]
			other_max = np.max(
				np.vstack([class_means[c] for c in classes if c != cls]), axis=0
			) if len(classes) > 1 else np.zeros_like(this_mean)

			# Positive margin favors tokens truly specific to this class.
			specificity = this_mean - other_max
			top_idx = np.argsort(specificity)[::-1]

			keywords: List[str] = []
			for idx in top_idx:
				token = feature_names[idx]
				if this_mean[idx] <= 0:
					continue
				if specificity[idx] <= 0:
					continue
				keywords.append(token)
				if len(keywords) >= self.top_k:
					break

			self.class_keywords[str(cls)] = keywords

		self._compile_patterns()
		return self

	def _compile_patterns(self) -> None:
		self._class_patterns = {}
		for cls, keywords in self.class_keywords.items():
			patterns: List[re.Pattern[str]] = []
			for kw in keywords:
				token_parts = [re.escape(part) for part in kw.split()]
				pattern = r"\b" + r"\s+".join(token_parts) + r"\b"
				patterns.append(re.compile(pattern, flags=re.IGNORECASE))
			self._class_patterns[cls] = patterns

	def predict(self, text: str, threshold: float = 0.90) -> Optional[LabelPrediction]:
		if not self._class_patterns:
			return None

		normalized_text = normalize_ticket_text(text)

		scores: Dict[str, float] = {}
		for cls, patterns in self._class_patterns.items():
			if not patterns:
				scores[cls] = 0.0
				continue

			matched = 0
			for pattern in patterns:
				if pattern.search(normalized_text):
					matched += 1

			# Rule confidence = matched keywords for class / total manifest keywords for class.
			scores[cls] = float(matched) / float(len(patterns))

		best_cls = max(scores, key=scores.get)
		best_score = float(scores[best_cls])
		if best_score <= 0.0:
			return None

		confidence = best_score
		abstain_flag = confidence < threshold
		routing_action: Literal["auto_route", "escalate_to_tier2"]
		routing_action = "escalate_to_tier2" if abstain_flag else "auto_route"

		return LabelPrediction(
			predicted_label=best_cls,
			confidence=confidence,
			tier_used="rules",
			routing_action=routing_action,
			abstain_flag=abstain_flag,
		)


class CalibratedLogReg:
	"""TF-IDF + class-balanced Logistic Regression with isotonic calibration."""

	def __init__(
		self,
		cv: int = 5,
		ngram_range: Tuple[int, int] = (1, 2),
		char_ngram_range: Tuple[int, int] = (3, 5),
		random_state: int = 42,
	) -> None:
		calibration_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
		combined_vectorizer = FeatureUnion(
			transformer_list=[
				("word_ngrams", TfidfVectorizer(analyzer="word", ngram_range=ngram_range)),
				("char_ngrams", TfidfVectorizer(analyzer="char", ngram_range=char_ngram_range)),
			]
		)
		self.model = CalibratedClassifierCV(
			estimator=Pipeline(
				steps=[
					("tfidf", combined_vectorizer),
					("logreg", LogisticRegression(class_weight="balanced", max_iter=1500)),
				]
			),
			method="isotonic",
			cv=calibration_cv,
		)
		self.classes_: Optional[np.ndarray] = None

	def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "CalibratedLogReg":
		self.model.fit(texts, labels)
		self.classes_ = np.asarray(self.model.classes_)
		return self

	def predict(self, text: str, threshold: float = 0.90) -> LabelPrediction:
		if self.classes_ is None:
			raise RuntimeError("LogReg model is not fitted. Call fit before predict.")

		normalized_text = normalize_ticket_text(text)
		probs = self.model.predict_proba([normalized_text])[0]
		idx = int(np.argmax(probs))
		label = str(self.classes_[idx])
		conf = float(probs[idx])
		abstain_flag = conf < threshold
		routing_action: Literal["auto_route", "escalate_to_tier2"]
		routing_action = "escalate_to_tier2" if abstain_flag else "auto_route"

		return LabelPrediction(
			predicted_label=label,
			confidence=conf,
			tier_used="logreg",
			routing_action=routing_action,
			abstain_flag=abstain_flag,
		)


class LabelHead:
	def __init__(
		self,
		threshold: float = 0.90,
		top_k_keywords: int = 10,
	) -> None:
		self.threshold = threshold
		self.rules = DataDrivenRules(top_k=top_k_keywords)
		self.logreg = CalibratedLogReg()
		self._is_fitted = False

	def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "LabelHead":
		normalized_texts = [normalize_ticket_text(text) for text in texts]
		self.rules.fit(normalized_texts, labels)
		self.logreg.fit(normalized_texts, labels)
		self._is_fitted = True
		return self

	def predict(self, text: str) -> LabelPrediction:
		if not self._is_fitted:
			raise RuntimeError("LabelHead is not fitted.")

		normalized_text = normalize_ticket_text(text)
		rule_hit = self.rules.predict(normalized_text, threshold=self.threshold)
		logreg_hit = self.logreg.predict(normalized_text, threshold=self.threshold)

		if rule_hit is None or logreg_hit.confidence >= rule_hit.confidence:
			chosen = logreg_hit
		else:
			chosen = rule_hit

		chosen.abstain_flag = chosen.confidence < self.threshold
		chosen.routing_action = "escalate_to_tier2" if chosen.abstain_flag else "auto_route"
		return chosen


class Tier1Router:
	"""Tier-1 hybrid classifier for Category and Priority."""

	def __init__(self, threshold: float = 0.90, top_k_keywords: int = 10, rule_min_confidence: float = 0.60) -> None:
		self.threshold = threshold
		self.rule_min_confidence = rule_min_confidence  # Backward-compatible constructor argument.
		self.priority_levels = {"low", "medium", "high", "critical"}
		self.categories = {"Other"}
		self.category_head = LabelHead(
			threshold=threshold,
			top_k_keywords=top_k_keywords,
		)
		self.priority_head = LabelHead(
			threshold=threshold,
			top_k_keywords=top_k_keywords,
		)
		self._is_fitted = False

	def fit(
		self,
		texts: Sequence[str],
		category_labels: Sequence[str],
		priority_labels: Sequence[str],
	) -> "Tier1Router":
		normalized_categories = [
			str(label).strip() if str(label).strip() else "Other"
			for label in category_labels
		]
		normalized_priorities = []
		for label in priority_labels:
			value = str(label).strip().lower()
			normalized_priorities.append(value if value in self.priority_levels else "medium")

		self.category_head.fit(texts, normalized_categories)
		self.priority_head.fit(texts, normalized_priorities)
		if self.category_head.logreg.classes_ is not None:
			learned_categories = {str(label) for label in self.category_head.logreg.classes_}
			self.categories = learned_categories.union({"Other"})
		self._is_fitted = True
		return self

	@classmethod
	def from_pretrained(
		cls,
		category_model_path: str,
		priority_model_path: str,
		threshold: float = 0.90,
	) -> "Tier1Router":
		"""Build a ready-to-serve router from pretrained Kaggle artifacts."""
		router = cls(threshold=threshold)
		router.category_head = load_head(category_model_path)
		router.priority_head = load_head(priority_model_path)
		router._is_fitted = True

		if router.category_head.logreg.classes_ is not None:
			router.categories = {str(label) for label in router.category_head.logreg.classes_}.union({"Other"})

		if router.priority_head.logreg.classes_ is not None:
			router.priority_levels = {str(label).lower() for label in router.priority_head.logreg.classes_}

		return router

	def predict_with_details(self, text: str) -> Dict[str, object]:
		if not self._is_fitted:
			raise RuntimeError("Tier1Router is not fitted.")

		normalized_text = normalize_ticket_text(text)
		category = self.category_head.predict(normalized_text)
		priority = self.priority_head.predict(normalized_text)

		category_label = category.predicted_label
		if category_label not in self.categories:
			category_label = "Other"

		priority_label = priority.predicted_label.lower()
		if priority_label not in self.priority_levels:
			priority_label = "medium"

		abstain_flag = category.abstain_flag or priority.abstain_flag
		overall_confidence = float(min(category.confidence, priority.confidence))

		return {
			"category": {
				"predicted_label": category_label,
				"confidence": category.confidence,
				"tier_used": category.tier_used,
				"routing_action": category.routing_action,
				"abstain_flag": category.abstain_flag,
			},
			"priority": {
				"predicted_label": priority_label,
				"confidence": priority.confidence,
				"tier_used": priority.tier_used,
				"routing_action": priority.routing_action,
				"abstain_flag": priority.abstain_flag,
			},
			"confidence_score": overall_confidence,
			"tier_used": category.tier_used if category.confidence >= priority.confidence else priority.tier_used,
			"abstain_flag": abstain_flag,
			"routing_action": "escalate_to_tier2" if abstain_flag else "auto_route",
		}

	def predict(self, text: str) -> TicketPrediction:
		details = self.predict_with_details(text)
		route = "auto_route" if not bool(details["abstain_flag"]) else "escalate_to_tier2"

		return TicketPrediction(
			predicted_category=str(details["category"]["predicted_label"]),
			predicted_priority=str(details["priority"]["predicted_label"]),
			confidence_score=float(details["confidence_score"]),
			tier_used=str(details["tier_used"]),
			abstain_flag=bool(details["abstain_flag"]),
			routing_action=route,
		)


def save_head(head: LabelHead, model_path: str) -> None:
	if not head._is_fitted:
		raise RuntimeError("Cannot save an unfitted model head.")
	joblib.dump(head, model_path)


def load_head(model_path: str) -> LabelHead:
	obj = joblib.load(model_path)
	if isinstance(obj, LabelHead):
		if not obj._is_fitted:
			raise RuntimeError("Loaded head is not fitted.")
		return obj

	# Handle sklearn Pipeline / classifier objects saved by model_training.py.
	# The training script saves plain sklearn Pipeline (TF-IDF + LogReg) via
	# pickle.dump().  Wrap them in a LabelHead so Tier1Router.from_pretrained()
	# can consume them.
	if hasattr(obj, "predict_proba") and hasattr(obj, "classes_"):
		head = LabelHead.__new__(LabelHead)
		head.threshold = 0.90
		head.rules = DataDrivenRules()          # unfitted — predict() returns None
		head.logreg = CalibratedLogReg.__new__(CalibratedLogReg)
		head.logreg.model = obj                 # sklearn Pipeline acts as the model
		head.logreg.classes_ = np.asarray(obj.classes_)
		head._is_fitted = True
		return head

	raise TypeError(
		f"Loaded object is not a LabelHead or compatible sklearn classifier: "
		f"{type(obj).__name__}"
	)
