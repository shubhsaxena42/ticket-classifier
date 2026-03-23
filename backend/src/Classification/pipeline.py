from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from Classification.tier_1 import Tier1Router, normalize_ticket_text
from Classification.tier_2 import Tier2Classifier
from Classification.tier_3 import groq_tier3_node
from RAG.HyDe import hyde_node
from RAG.retrieval import TwoStageRetriever, load_retrieval_components
from Scrubber.scrub import run_scrub_ticket
from generation.abstention_gate import ABSTAIN_RESPONSE_TEXT, AbstentionGate
from generation.generator import ResponseGenerator
from generation.schemas import TicketPrediction as GenerationTicketPrediction
from Scrubber.Rules import TicketPrediction


@dataclass
class _Tier1BaselineBundle:
	category_model: object
	priority_model: object
	threshold: float

	def predict(self, text: str) -> TicketPrediction:
		cat_probs = self.category_model.predict_proba([text])[0]
		cat_idx = int(cat_probs.argmax())
		cat_label = str(self.category_model.classes_[cat_idx])
		cat_conf = float(cat_probs[cat_idx])

		pri_probs = self.priority_model.predict_proba([text])[0]
		pri_idx = int(pri_probs.argmax())
		pri_label = str(self.priority_model.classes_[pri_idx]).lower()
		pri_conf = float(pri_probs[pri_idx])

		confidence = float(min(cat_conf, pri_conf))
		abstain = confidence < self.threshold
		route = "escalate_to_tier2" if abstain else "auto_route"

		return TicketPrediction(
			predicted_category=cat_label,
			predicted_priority=pri_label,
			confidence_score=confidence,
			tier_used="logreg",
			abstain_flag=abstain,
			routing_action=route,
		)


class SupportCascadePipeline:
	"""End-to-end support pipeline: scrub -> classify -> retrieve -> abstain -> generate."""

	def __init__(
		self,
		tier1_router: Optional[Tier1Router],
		tier1_baseline: Optional[_Tier1BaselineBundle],
		tier2_classifier: Tier2Classifier,
		two_stage_retriever: TwoStageRetriever,
		abstention_gate: AbstentionGate,
		response_generator: ResponseGenerator,
		enable_tier3_llm: bool = True,
		kb_path: Optional[Path] = None,
	) -> None:
		if tier1_router is None and tier1_baseline is None:
			raise ValueError("A Tier 1 model must be provided.")
		self.tier1_router = tier1_router
		self.tier1_baseline = tier1_baseline
		self.tier2_classifier = tier2_classifier
		self.two_stage_retriever = two_stage_retriever
		self.abstention_gate = abstention_gate
		self.response_generator = response_generator
		self.enable_tier3_llm = bool(enable_tier3_llm)
		self.kb_path = kb_path

	@staticmethod
	def _resolve_path(base_dir: Path, rel_or_abs: str) -> Path:
		candidate = Path(rel_or_abs)
		if candidate.is_absolute():
			return candidate
		return (base_dir / rel_or_abs).resolve()

	@classmethod
	def from_manifest(
		cls,
		manifest_path: str = "results/training_manifest.json",
		enable_tier3_llm: bool = True,
	) -> "SupportCascadePipeline":
		manifest_file = Path(manifest_path).resolve()
		if not manifest_file.exists():
			raise FileNotFoundError(f"Manifest not found: {manifest_file}")

		manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
		base_dir = manifest_file.parent
		artifacts: Dict[str, Any] = manifest.get("artifacts", {})
		thresholds: Dict[str, Any] = manifest.get("thresholds", {})

		tier1_threshold = float(thresholds.get("tier1_confidence", 0.90))
		tier2_threshold = float(thresholds.get("tier2_confidence", 0.45))

		tier1_router: Optional[Tier1Router] = None
		tier1_baseline: Optional[_Tier1BaselineBundle] = None

		# Preferred: custom tier1 heads saved from this codebase.
		custom_cat_key = artifacts.get("tier1_category_head") or artifacts.get("tier1_cat_head")
		custom_pri_key = artifacts.get("tier1_priority_head") or artifacts.get("tier1_pri_head")
		if custom_cat_key and custom_pri_key:
			cat_path = cls._resolve_path(base_dir, str(custom_cat_key))
			pri_path = cls._resolve_path(base_dir, str(custom_pri_key))
			if cat_path.exists() and pri_path.exists():
				tier1_router = Tier1Router.from_pretrained(
					category_model_path=str(cat_path),
					priority_model_path=str(pri_path),
					threshold=tier1_threshold,
				)

		# Fallback: baseline category/priority sklearn pipelines from manifest.
		if tier1_router is None:
			baseline_cat = artifacts.get("baseline_cat")
			baseline_pri = artifacts.get("baseline_pri")
			if not baseline_cat or not baseline_pri:
				raise FileNotFoundError("Tier 1 pretrained artifacts not found in manifest.")

			baseline_cat_path = cls._resolve_path(base_dir, str(baseline_cat))
			baseline_pri_path = cls._resolve_path(base_dir, str(baseline_pri))
			if not baseline_cat_path.exists() or not baseline_pri_path.exists():
				raise FileNotFoundError(
					f"Tier 1 baseline artifacts missing: {baseline_cat_path}, {baseline_pri_path}"
				)

			tier1_baseline = _Tier1BaselineBundle(
				category_model=joblib.load(baseline_cat_path),
				priority_model=joblib.load(baseline_pri_path),
				threshold=tier1_threshold,
			)

		setfit_cat = artifacts.get("setfit_category")
		setfit_pri = artifacts.get("setfit_priority")
		calib_cat = artifacts.get("calibration_cat")
		calib_pri = artifacts.get("calibration_pri")
		if not all([setfit_cat, setfit_pri, calib_cat, calib_pri]):
			raise FileNotFoundError("Tier 2 pretrained artifact paths are missing in manifest.")

		tier2 = Tier2Classifier.from_pretrained(
			category_model_dir=str(cls._resolve_path(base_dir, str(setfit_cat))),
			priority_model_dir=str(cls._resolve_path(base_dir, str(setfit_pri))),
			category_calibration_path=str(cls._resolve_path(base_dir, str(calib_cat))),
			priority_calibration_path=str(cls._resolve_path(base_dir, str(calib_pri))),
			gate_threshold=tier2_threshold,
		)

		two_stage_retriever = load_retrieval_components(
			models_dir=base_dir / "models",
			data_dir=base_dir / "processed",
		)

		abstention_gate = AbstentionGate(threshold=0.35)

		response_generator = ResponseGenerator()

		kb_default = Path(__file__).resolve().parents[2] / "Data" / "Raw_Data" / "kb.md"
		return cls(
			tier1_router=tier1_router,
			tier1_baseline=tier1_baseline,
			tier2_classifier=tier2,
			two_stage_retriever=two_stage_retriever,
			abstention_gate=abstention_gate,
			response_generator=response_generator,
			enable_tier3_llm=enable_tier3_llm,
			kb_path=kb_default,
		)

	@staticmethod
	def _apply_operating_point(prediction: TicketPrediction) -> TicketPrediction:
		"""Map final confidence to routing action: >=0.90 auto_route, 0.65-0.90 suggest, <0.65 manual_review."""
		conf = float(prediction.confidence_score or 0.0)
		if conf >= 0.90:
			prediction.routing_action = "auto_route"
		elif conf >= 0.65:
			prediction.routing_action = "suggest"
		else:
			prediction.routing_action = "manual_review"
		return prediction

	def _predict_tier1(self, cleaned_text: str) -> TicketPrediction:
		if self.tier1_router is not None:
			return self.tier1_router.predict(cleaned_text)
		if self.tier1_baseline is not None:
			return self.tier1_baseline.predict(cleaned_text)
		raise RuntimeError("Tier 1 model is not initialized.")

	def _predict_classification(self, cleaned_text: str) -> TicketPrediction:
		normalized_text = normalize_ticket_text(cleaned_text)

		tier1_prediction = self._predict_tier1(normalized_text)
		if tier1_prediction.routing_action == "auto_route" and not bool(tier1_prediction.abstain_flag):
			return self._apply_operating_point(tier1_prediction)

		tier2_prediction = self.tier2_classifier.predict(normalized_text)
		if tier2_prediction.routing_action == "auto_route" and not bool(tier2_prediction.abstain_flag):
			return self._apply_operating_point(tier2_prediction)

		if not self.enable_tier3_llm:
			fallback = TicketPrediction(
				predicted_category=tier2_prediction.predicted_category,
				predicted_priority=tier2_prediction.predicted_priority,
				confidence_score=tier2_prediction.confidence_score,
				tier_used=tier2_prediction.tier_used,
				abstain_flag=True,
				routing_action="manual_review",
			)
			return self._apply_operating_point(fallback)

		tier3_prediction = groq_tier3_node(cleaned_text=normalized_text, kb_path=self.kb_path)
		return self._apply_operating_point(tier3_prediction)

	def _run_grounded_generation(self, cleaned_message: str) -> Dict[str, Any]:
		state: Dict[str, Any] = {
			"hyde_queries": [cleaned_message],
			"reranked_chunks": [],
			"abstain_flag": True,
			"draft_response": ABSTAIN_RESPONSE_TEXT,
			"citations": [],
			"top_3_sources": [],
			"retrieval_max_score": 0.0,
		}

		if not cleaned_message.strip():
			return state  # empty text after normalisation — skip RAG, return abstain

		try:
			hyde_update = hyde_node({"cleaned_text": cleaned_message})
		except ValueError:
			return state  # HyDE requires non-empty text; skip RAG
		queries = hyde_update.get("queries", [])
		if isinstance(queries, list) and queries:
			state["hyde_queries"] = [str(q) for q in queries]

		rerank_update = self.two_stage_retriever.retrieve_for_langgraph(
			{"hyde_queries": state["hyde_queries"]}
		)
		reranked_chunks = rerank_update.get("reranked_chunks", [])
		if isinstance(reranked_chunks, list):
			state["reranked_chunks"] = reranked_chunks

		gate_update = self.abstention_gate.check_for_langgraph(
			{"reranked_chunks": state["reranked_chunks"]}
		)
		state.update(gate_update)

		if not bool(state.get("abstain_flag", True)):
			gen_update = self.response_generator.generate_for_langgraph(
				{
					"cleaned_message": cleaned_message,
					"reranked_chunks": state["reranked_chunks"],
				}
			)
			state.update(gen_update)

		if not state.get("top_3_sources"):
			state["top_3_sources"] = [
				str(chunk.get("source", ""))
				for chunk in state.get("reranked_chunks", [])
			]

		if not state.get("retrieval_max_score"):
			max_score = 0.0
			for chunk in state.get("reranked_chunks", []):
				score = float(chunk.get("rerank_score", 0.0))
				max_score = max(max_score, score)
			state["retrieval_max_score"] = max_score

		return state

	def predict_text(self, cleaned_text: str) -> TicketPrediction:
		return self._predict_classification(cleaned_text)

	def predict_ticket_state(self, ticket: Dict[str, Any], ticket_id: Optional[str] = None) -> Dict[str, Any]:
		scrubbed_ticket, _ = run_scrub_ticket(ticket=ticket, include_resolution=False)
		# Subject excluded: model trained on description only (subject leaks category label)
		description = str(scrubbed_ticket.get("Ticket Description", "") or "")
		cleaned_text = normalize_ticket_text(description)

		classification_prediction = self._predict_classification(cleaned_text)

		state: Dict[str, Any] = {
			"ticket_id": str(ticket_id or ticket.get("ticket_id") or ticket.get("Ticket ID") or ""),
			"cleaned_message": cleaned_text,
			"predicted_category": classification_prediction.predicted_category,
			"predicted_priority": classification_prediction.predicted_priority,
			"confidence_score": float(classification_prediction.confidence_score or 0.0),
			"tier_used": str(classification_prediction.tier_used or ""),
			"routing_action": str(classification_prediction.routing_action),
			"abstain_flag": bool(classification_prediction.abstain_flag or False),
		}

		rag_state = self._run_grounded_generation(cleaned_message=cleaned_text)
		state.update(rag_state)
		return state

	def predict_ticket_record(self, ticket: Dict[str, Any], ticket_id: Optional[str] = None) -> GenerationTicketPrediction:
		state = self.predict_ticket_state(ticket=ticket, ticket_id=ticket_id)
		resolved_ticket_id = str(state.get("ticket_id", "") or ticket_id or "")
		return GenerationTicketPrediction.from_state(state=state, ticket_id=resolved_ticket_id)

	def predict_ticket(self, ticket: Dict[str, Any]) -> TicketPrediction:
		state = self.predict_ticket_state(ticket=ticket)
		return TicketPrediction(
			predicted_category=str(state.get("predicted_category", "")),
			predicted_priority=str(state.get("predicted_priority", "")),
			confidence_score=float(state.get("confidence_score", 0.0)),
			tier_used=str(state.get("tier_used", "")),
			abstain_flag=bool(state.get("abstain_flag", False)),
			routing_action=str(state.get("routing_action", "manual_review")),
		)

	def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
		rows: List[Dict[str, Any]] = []
		for _, row in df.iterrows():
			ticket = row.to_dict()
			ticket_id = ticket.get("ticket_id") or ticket.get("Ticket ID") or ""
			prediction = self.predict_ticket_record(ticket=ticket, ticket_id=str(ticket_id))
			rows.append(
				{
					**ticket,
					"predicted_category": prediction.predicted_category,
					"predicted_priority": prediction.predicted_priority,
					"top_3_retrieved_sources": prediction.top_3_retrieved_sources,
					"draft_response": prediction.draft_response,
					"confidence_score": prediction.confidence_score,
					"citations": prediction.citations,
					"tier_used": prediction.tier_used,
					"abstain_flag": prediction.abstain_flag,
					"routing_action": prediction.routing_action,
					"retrieval_max_score": prediction.retrieval_max_score,
				}
			)
		return pd.DataFrame(rows)
