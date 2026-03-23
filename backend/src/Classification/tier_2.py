from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from Scrubber.Rules import TicketPrediction

# Delay SetFit imports to runtime and prefer low-level modules used for inference.
# This avoids importing setfit.model_card on environments where huggingface_hub
# removed DatasetFilter.
SetFitModel = None
SetFitTrainer = None
Dataset = None

def _require_setfit() -> None:
	global SetFitModel, SetFitTrainer, Dataset

	# SetFit imports DatasetFilter from huggingface_hub.model_card paths.
	# Newer hub versions may not expose this symbol; add a no-op shim so
	# inference can proceed when model-card functionality is unused.
	try:
		import huggingface_hub as _hfhub
		if not hasattr(_hfhub, "DatasetFilter"):
			class _DatasetFilter:  # pragma: no cover - compatibility shim
				def __init__(self, *args, **kwargs):
					self.args = args
					self.kwargs = kwargs

			_hfhub.DatasetFilter = _DatasetFilter
	except Exception:
		pass

	# setfit==1.0.3 bug: when loading a locally-saved model, config.json may
	# not contain '_name_or_path', causing config_dict.get('_name_or_path') to
	# return None.  Path(None) then raises TypeError.  Patch infer_st_id to
	# skip gracefully when the field is missing.
	try:
		from setfit.model_card import SetFitModelCardData as _CardData
		from pathlib import Path as _Path

		def _safe_infer_st_id(self, setfit_model_id: str) -> None:  # type: ignore[override]
			try:
				from transformers import PretrainedConfig
				config_dict, _ = PretrainedConfig.get_config_dict(setfit_model_id)
				st_id = config_dict.get("_name_or_path")
				if not st_id:  # None or empty string — local model, nothing to infer
					return
				st_id_path = _Path(st_id)
				# Replicate the rest of the original logic safely
				if st_id_path.exists():
					return
				if st_id == setfit_model_id:
					return
				self.st_id = st_id
			except Exception:
				pass  # Never block inference over a model-card metadata failure

		if not getattr(_CardData, "_patched_infer_st_id", False):
			_CardData.infer_st_id = _safe_infer_st_id
			_CardData._patched_infer_st_id = True  # type: ignore[attr-defined]
	except Exception:
		pass

	if SetFitModel is None:
		try:
			from setfit.modeling import SetFitModel as _SetFitModel
		except Exception:
			from setfit import SetFitModel as _SetFitModel
		SetFitModel = _SetFitModel

	if SetFitTrainer is None:
		try:
			from setfit import SetFitTrainer as _SetFitTrainer
			SetFitTrainer = _SetFitTrainer
		except Exception:
			SetFitTrainer = None

	if Dataset is None:
		try:
			from datasets import Dataset as _Dataset
			Dataset = _Dataset
		except Exception:
			Dataset = None


def _compose_text(df: pd.DataFrame) -> pd.Series:
	# Subject excluded: model trained on description only (subject leaks category label)
	description = df.get("Ticket Description", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
	return description.str.lower().str.strip()


def _safe_texts(texts: Iterable[str]) -> list[str]:
	return [str(text).strip() for text in texts]


def _encode_texts(model: object, texts: Sequence[str]) -> np.ndarray:
	if hasattr(model, "encode"):
		return np.asarray(model.encode(list(texts)))
	if hasattr(model, "model_body") and hasattr(model.model_body, "encode"):
		return np.asarray(model.model_body.encode(list(texts)))
	raise AttributeError("SetFit model does not expose an encode method.")


def _nll_with_temperature(logits: np.ndarray, y_true: np.ndarray, temperature: float) -> float:
	temp = float(max(temperature, 1e-6))
	scaled = logits / temp

	if scaled.ndim == 1:
		probs_pos = expit(scaled)
		probs_pos = np.clip(probs_pos, 1e-9, 1.0 - 1e-9)
		return float(-np.mean(y_true * np.log(probs_pos) + (1 - y_true) * np.log(1.0 - probs_pos)))

	probs = softmax(scaled, axis=1)
	probs = np.clip(probs, 1e-9, 1.0)
	rows = np.arange(len(y_true))
	return float(-np.mean(np.log(probs[rows, y_true])))


class TemperatureScaledLogReg(BaseEstimator, ClassifierMixin):
	"""A frozen logistic head with temperature scaling in predict_proba."""

	def __init__(self, head: LogisticRegression, temperature: float = 1.0) -> None:
		self.head = head
		self.temperature = float(max(temperature, 1e-6))
		self.classes_ = np.asarray(head.classes_)

	def fit(self, x: np.ndarray, y: Sequence[str]):  # noqa: D401 - sklearn style
		# Frozen estimator: fit is a no-op to keep compatibility with calibrators.
		return self

	def _scaled_logits(self, x: np.ndarray) -> np.ndarray:
		logits = self.head.decision_function(x)
		return np.asarray(logits) / self.temperature

	def predict_proba(self, x: np.ndarray) -> np.ndarray:
		scaled = self._scaled_logits(x)
		if scaled.ndim == 1:
			pos = expit(scaled)
			return np.vstack([1.0 - pos, pos]).T
		return softmax(scaled, axis=1)

	def predict(self, x: np.ndarray) -> np.ndarray:
		probs = self.predict_proba(x)
		idx = np.argmax(probs, axis=1)
		return self.classes_[idx]


@dataclass
class _SpecialistBundle:
	model: object
	head: LogisticRegression
	temperature: float = 1.0
	calibrator: Optional[CalibratedClassifierCV] = None


class Tier2Classifier:
	"""Tier-2 SetFit specialist with post-training calibration and gate logic."""

	def __init__(
		self,
		base_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
		gate_threshold: float = 0.45,
		num_iterations: int = 20,
		num_epochs: int = 1,
		batch_size: int = 16,
	) -> None:
		self.base_model_id = base_model_id
		self.gate_threshold = gate_threshold
		self.num_iterations = num_iterations
		self.num_epochs = num_epochs
		self.batch_size = batch_size

		self._category: Optional[_SpecialistBundle] = None
		self._priority: Optional[_SpecialistBundle] = None
		self._is_trained = False

	@staticmethod
	def _load_bundle(model_dir: str, calibration_path: str) -> _SpecialistBundle:
		_require_setfit()
		model = SetFitModel.from_pretrained(model_dir)
		payload = joblib.load(calibration_path)

		# ── Resolve the LogisticRegression head ───────────────────────────────
		# Priority order:
		#   1. Explicit "head" key in the calibration pkl  (save_pretrained format)
		#   2. model.model_head                            (SetFit stores it inside model dir)
		#   3. Unwrap from a CalibratedClassifierCV stored in the pkl
		#   4. The payload object itself if it has predict_proba
		head = None
		temperature = 1.0
		calibrator = None  # will hold either CalibratedClassifierCV OR the custom calib dict

		if isinstance(payload, dict):
			head = payload.get("head")
			temperature = float(payload.get("temperature", 1.0))

			# ── Custom two-stage calibration format ──────────────────────────
			# Keys: "calibrators" (dict class→IsotonicRegression),
			#       "classes", "temperature", "use_temperature", "use_isotonic"
			if "calibrators" in payload and head is None:
				# Head lives inside the SetFit model directory as model_head.pkl
				if hasattr(model, "model_head") and model.model_head is not None:
					head = model.model_head
					if hasattr(head, "head"):  # TemperatureScaledLogReg wrapper
						head = head.head
				# Store the full calibration payload as the calibrator so
				# _predict_specialist can apply temperature + isotonic stages.
				calibrator = payload  # custom dict, handled in _predict_specialist

			# ── sklearn CalibratedClassifierCV stored under "calibrator" key ──
			sklearn_cal = payload.get("calibrator")
			if head is None and sklearn_cal is not None:
				calibrator = sklearn_cal
				if hasattr(sklearn_cal, "estimator"):
					head = sklearn_cal.estimator
					if hasattr(head, "head"):
						head = head.head
				elif hasattr(sklearn_cal, "base_estimator"):
					head = sklearn_cal.base_estimator
		else:
			# Payload IS the estimator directly (legacy format)
			if hasattr(payload, "estimator"):
				calibrator = payload
				head = payload.estimator
				if hasattr(head, "head"):
					head = head.head
			elif hasattr(payload, "predict_proba"):
				head = payload

		# Final fallback: grab head directly from the SetFit model
		if head is None and hasattr(model, "model_head") and model.model_head is not None:
			head = model.model_head
			if hasattr(head, "head"):
				head = head.head

		if head is None:
			raise ValueError(
				f"Cannot resolve a LogisticRegression head from artifacts.\n"
				f"Calibration pkl type: {type(payload).__name__}\n"
				f"Calibration pkl keys: {list(payload.keys()) if isinstance(payload, dict) else 'N/A'}\n"
				f"SetFit model_head: {type(getattr(model, 'model_head', None)).__name__}"
			)


		return _SpecialistBundle(model=model, head=head, temperature=temperature, calibrator=calibrator)

	@classmethod
	def from_pretrained(
		cls,
		category_model_dir: str,
		priority_model_dir: str,
		category_calibration_path: str,
		priority_calibration_path: str,
		gate_threshold: float = 0.45,
	) -> "Tier2Classifier":
		"""Load a ready-to-serve Tier 2 stack (no local training required)."""
		instance = cls(gate_threshold=gate_threshold)
		instance._category = cls._load_bundle(category_model_dir, category_calibration_path)
		instance._priority = cls._load_bundle(priority_model_dir, priority_calibration_path)
		instance._is_trained = True
		return instance

	def save_pretrained(
		self,
		category_model_dir: str,
		priority_model_dir: str,
		category_calibration_path: str,
		priority_calibration_path: str,
	) -> None:
		"""Persist Tier 2 artifacts so they can be loaded via from_pretrained()."""
		if not self._is_trained or self._category is None or self._priority is None:
			raise RuntimeError("Tier2Classifier is not ready. Train/calibrate before saving artifacts.")

		if hasattr(self._category.model, "save_pretrained"):
			self._category.model.save_pretrained(category_model_dir)
		else:
			raise AttributeError("Category SetFit model does not support save_pretrained().")

		if hasattr(self._priority.model, "save_pretrained"):
			self._priority.model.save_pretrained(priority_model_dir)
		else:
			raise AttributeError("Priority SetFit model does not support save_pretrained().")

		joblib.dump(
			{
				"head": self._category.head,
				"temperature": self._category.temperature,
				"calibrator": self._category.calibrator,
			},
			category_calibration_path,
		)
		joblib.dump(
			{
				"head": self._priority.head,
				"temperature": self._priority.temperature,
				"calibrator": self._priority.calibrator,
			},
			priority_calibration_path,
		)

	def _train_setfit_specialist(self, texts: Sequence[str], labels: Sequence[str]) -> _SpecialistBundle:
		_require_setfit()
		model = SetFitModel.from_pretrained(self.base_model_id)

		if SetFitTrainer is not None and Dataset is not None:
			train_ds = Dataset.from_dict({"text": list(texts), "label": list(labels)})
			trainer = SetFitTrainer(
				model=model,
				train_dataset=train_ds,
				num_iterations=self.num_iterations,
				num_epochs=self.num_epochs,
				batch_size=self.batch_size,
				column_mapping={"text": "text", "label": "label"},
			)
			trainer.train()
		elif hasattr(model, "fit"):
			model.fit(
				x_train=list(texts),
				y_train=list(labels),
				num_iterations=self.num_iterations,
				num_epochs=self.num_epochs,
				batch_size=self.batch_size,
			)
		else:
			raise RuntimeError("SetFit training API is unavailable in this environment.")

		embeddings = _encode_texts(model, texts)
		head = LogisticRegression(max_iter=1000)
		head.fit(embeddings, list(labels))
		return _SpecialistBundle(model=model, head=head)

	def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> "Tier2Classifier":
		texts = _safe_texts(_compose_text(train_df))
		category_labels = [str(v).strip() for v in train_df["Ticket Type"].fillna("Other")]
		priority_labels = [str(v).strip().lower() for v in train_df["Ticket Priority"].fillna("medium")]

		self._category = self._train_setfit_specialist(texts, category_labels)
		self._priority = self._train_setfit_specialist(texts, priority_labels)
		self._is_trained = True

		# Calibration requires validation labels and trained specialists.
		self.calibrate(val_df)
		return self

	def _calibrate_specialist(
		self,
		specialist: _SpecialistBundle,
		texts: Sequence[str],
		labels: Sequence[str],
	) -> None:
		embeddings = _encode_texts(specialist.model, texts)
		classes = np.asarray(specialist.head.classes_)
		y_idx = np.searchsorted(classes, np.asarray(labels))

		logits = np.asarray(specialist.head.decision_function(embeddings))

		def _objective(temp_arr: np.ndarray) -> float:
			return _nll_with_temperature(logits, y_idx, float(temp_arr[0]))

		result = minimize(_objective, x0=np.array([1.0]), bounds=[(1e-3, 20.0)], method="L-BFGS-B")
		temperature = float(result.x[0]) if result.success else 1.0
		specialist.temperature = max(temperature, 1e-3)

		scaled_estimator = TemperatureScaledLogReg(specialist.head, specialist.temperature)
		calibrator = CalibratedClassifierCV(estimator=scaled_estimator, method="isotonic", cv=3)
		calibrator.fit(embeddings, np.asarray(labels))
		specialist.calibrator = calibrator

	def calibrate(self, val_df: pd.DataFrame) -> "Tier2Classifier":
		if self._category is None or self._priority is None:
			raise RuntimeError("Tier2Classifier must be trained before calibration.")

		texts = _safe_texts(_compose_text(val_df))
		category_labels = [str(v).strip() for v in val_df["Ticket Type"].fillna("Other")]
		priority_labels = [str(v).strip().lower() for v in val_df["Ticket Priority"].fillna("medium")]

		self._calibrate_specialist(self._category, texts, category_labels)
		self._calibrate_specialist(self._priority, texts, priority_labels)
		return self

	def _predict_specialist(self, specialist: _SpecialistBundle, text: str) -> Tuple[str, float]:
		embeddings = _encode_texts(specialist.model, [text])

		# ── Custom two-stage calibration: temperature scale → per-class isotonic ──
		if isinstance(specialist.calibrator, dict) and "calibrators" in specialist.calibrator:
			calib_payload = specialist.calibrator
			classes = np.asarray(calib_payload["classes"])

			# Step 1: get raw LogReg probabilities (temperature-scaled if needed)
			if hasattr(specialist.head, "predict_proba"):
				raw_probs = np.asarray(specialist.head.predict_proba(embeddings)[0])
			else:
				scaled = TemperatureScaledLogReg(specialist.head, specialist.temperature)
				raw_probs = scaled.predict_proba(embeddings)[0]

			# Step 2: apply temperature scaling to logits if flag is set
			if calib_payload.get("use_temperature", False):
				logits = np.log(np.clip(raw_probs, 1e-9, 1.0))
				temp = float(calib_payload.get("temperature", 1.0))
				raw_probs = softmax(logits / max(temp, 1e-6))

			# Step 3: apply per-class isotonic calibration if flag is set
			if calib_payload.get("use_isotonic", False):
				calibrators: dict = calib_payload["calibrators"]
				cal_probs = np.zeros(len(classes))
				for i, cls in enumerate(classes):
					iso = calibrators.get(str(cls))
					if iso is not None:
						cal_probs[i] = float(iso.predict([raw_probs[i]])[0])
					else:
						cal_probs[i] = raw_probs[i]
				# Re-normalise so probabilities sum to 1
				total = cal_probs.sum()
				probs = cal_probs / total if total > 1e-9 else cal_probs
			else:
				probs = raw_probs

		# ── sklearn CalibratedClassifierCV (older format) ─────────────────────
		elif specialist.calibrator is not None:
			probs = specialist.calibrator.predict_proba(embeddings)[0]
			classes = np.asarray(specialist.calibrator.classes_)

		# ── No calibrator: plain temperature-scaled LogReg ────────────────────
		else:
			scaled = TemperatureScaledLogReg(specialist.head, specialist.temperature)
			probs = scaled.predict_proba(embeddings)[0]
			classes = np.asarray(scaled.classes_)

		idx = int(np.argmax(probs))
		return str(classes[idx]), float(probs[idx])

	def predict(self, text: str) -> TicketPrediction:
		if not self._is_trained or self._category is None or self._priority is None:
			raise RuntimeError("Tier2Classifier is not ready. Call train()/calibrate() or from_pretrained().")

		cleaned_text = str(text).strip().lower()
		pred_cat, conf_cat = self._predict_specialist(self._category, cleaned_text)
		pred_pri, conf_pri = self._predict_specialist(self._priority, cleaned_text)

		calibrated_confidence = float(min(conf_cat, conf_pri))
		abstain_flag = calibrated_confidence < self.gate_threshold
		routing_action = "escalate_to_tier3" if abstain_flag else "auto_route"

		return TicketPrediction(
			predicted_category=pred_cat,
			predicted_priority=pred_pri,
			confidence_score=calibrated_confidence,
			tier_used="setfit",
			abstain_flag=abstain_flag,
			routing_action=routing_action,
		)
