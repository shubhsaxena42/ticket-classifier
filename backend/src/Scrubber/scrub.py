from __future__ import annotations

import argparse
import html
import re
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import spacy


# Shared preprocessing order for train/serve parity across the cascade.
# 1) PII scrub (regex -> NER), 2) normalize (lowercase, punctuation cleanup), 3) lemmatize.
PREPROCESSING_ORDER: Tuple[str, ...] = ("pii_scrub", "normalize", "lemmatize")


def _project_root() -> Path:
	resolved = Path(__file__).resolve()
	if len(resolved.parents) >= 3:
		return resolved.parents[2]
	return Path.cwd()


def _find_kaggle_input_csv(filename: str = "customer_support_tickets.csv") -> Optional[Path]:
	kaggle_input = Path("/kaggle/input")
	if not kaggle_input.exists():
		return None

	direct = kaggle_input / filename
	if direct.exists():
		return direct

	matches = sorted(kaggle_input.glob(f"**/{filename}"))
	if matches:
		return matches[0]
	return None


def default_input_path() -> Path:
	kaggle_csv = _find_kaggle_input_csv()
	if kaggle_csv is not None:
		return kaggle_csv

	root = _project_root()
	return root / "Data" / "Raw_Data" / "customer_support_tickets.csv"


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

PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
	(
		"EMAIL",
		re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
	),
	(
		"PHONE",
		re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"),
	),
	("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
	("ZIP", re.compile(r"\b\d{5}(?:-\d{4})?\b")),
	(
		"IP",
		re.compile(
			r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
		),
	),
	(
		"CREDIT_CARD",
		re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
	),
]


@dataclass
class ScrubStats:
	regex_matches: int = 0
	ner_matches: int = 0
	cells_changed: int = 0


class PIIScrubber:
	def __init__(self, model_name: str = "en_core_web_sm") -> None:
		self.model_name = model_name
		self.nlp = self._load_spacy_model(model_name)

	@staticmethod
	def _load_spacy_model(model_name: str):
		try:
			return spacy.load(model_name)
		except OSError as exc:
			raise RuntimeError(
				"spaCy model not found. Install it with: "
				"python -m spacy download en_core_web_sm"
			) from exc

	def _apply_regex_scrub(self, text: str, stats: ScrubStats) -> str:
		scrubbed = text
		for label, pattern in PATTERNS:
			replacement = REPLACEMENTS[label]

			def _replacer(match: re.Match[str]) -> str:
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
		# Replace from right to left so string offsets remain valid.
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

	def scrub_dataframe(
		self,
		df: pd.DataFrame,
		text_columns: Iterable[str],
		structured_columns: Dict[str, str],
	) -> Tuple[pd.DataFrame, ScrubStats]:
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

			def _scrub_cell(value: object) -> object:
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

	def scrub_ticket(
		self,
		ticket: Dict[str, object],
		text_columns: Iterable[str],
		structured_columns: Dict[str, str],
	) -> Tuple[Dict[str, object], ScrubStats]:
		"""Scrub a single ticket payload while preserving non-target fields."""
		df = pd.DataFrame([ticket])
		scrubbed_df, stats = self.scrub_dataframe(
			df=df,
			text_columns=text_columns,
			structured_columns=structured_columns,
		)
		return scrubbed_df.iloc[0].to_dict(), stats


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Scrub PII from a CSV using regex + spaCy NER.")
	parser.add_argument(
		"--input",
		type=Path,
		default=default_input_path(),
		help="Path to input CSV file.",
	)
	parser.add_argument(
		"--model",
		type=str,
		default="en_core_web_sm",
		help="spaCy model name for NER.",
	)
	parser.add_argument(
		"--exclude-resolution",
		action="store_true",
		help="Skip Resolution text scrubbing to reduce latency for online inference.",
	)
	parser.add_argument(
		"--drop-ticket-id",
		action="store_true",
		help="Drop Ticket ID from output. By default Ticket ID is preserved for traceability.",
	)
	return parser.parse_args()


def age_to_bucket(value: object, bucket_size: int = 5) -> object:
	if pd.isna(value):
		return value
	try:
		age = int(float(value))
	except (ValueError, TypeError):
		return value

	lower = math.floor(age / bucket_size) * bucket_size
	upper = lower + bucket_size
	return f"{lower}-{upper}"


def _build_text_columns(include_resolution: bool = True) -> List[str]:
	text_columns = [
		"Ticket Subject",
		"Ticket Description",
	]
	if include_resolution:
		text_columns.append("Resolution")
	return text_columns


def run_scrub(
	input_path: Path,
	model_name: str = "en_core_web_sm",
	include_resolution: bool = True,
	preserve_ticket_id: bool = True,
) -> Tuple[pd.DataFrame, ScrubStats]:
	df = pd.read_csv(input_path)
	if not preserve_ticket_id and "Ticket ID" in df.columns:
		df = df.drop(columns=["Ticket ID"])

	if "Customer Age" in df.columns:
		df["Customer Age"] = df["Customer Age"].apply(age_to_bucket)

	scrubber = PIIScrubber(model_name=model_name)

	# Direct identifiers are fully masked at column level.
	structured_columns = {
		"Customer Name": "[REDACTED_PERSON]",
		"Customer Email": "[REDACTED_EMAIL]",
	}

	# Free-text fields get regex + NER redaction.
	text_columns = _build_text_columns(include_resolution=include_resolution)

	scrubbed_df, stats = scrubber.scrub_dataframe(
		df=df,
		text_columns=text_columns,
		structured_columns=structured_columns,
	)

	return scrubbed_df, stats


def run_scrub_ticket(
	ticket: Dict[str, object],
	model_name: str = "en_core_web_sm",
	include_resolution: bool = False,
) -> Tuple[Dict[str, object], ScrubStats]:
	"""Scrub one ticket payload for low-latency API inference."""
	scrubber = PIIScrubber(model_name=model_name)
	structured_columns = {
		"Customer Name": "[REDACTED_PERSON]",
		"Customer Email": "[REDACTED_EMAIL]",
	}
	text_columns = _build_text_columns(include_resolution=include_resolution)

	ticket_copy = dict(ticket)
	if "Customer Age" in ticket_copy:
		ticket_copy["Customer Age"] = age_to_bucket(ticket_copy["Customer Age"])

	return scrubber.scrub_ticket(
		ticket=ticket_copy,
		text_columns=text_columns,
		structured_columns=structured_columns,
	)


def main() -> None:
	args = parse_args()
	scrubbed_df, stats = run_scrub(
		args.input,
		model_name=args.model,
		include_resolution=not args.exclude_resolution,
		preserve_ticket_id=not args.drop_ticket_id,
	)

	print(f"Input:  {args.input}")
	print(f"Rows:   {len(scrubbed_df)}")
	print(
		"Scrub summary -> "
		f"regex matches: {stats.regex_matches}, "
		f"NER matches: {stats.ner_matches}, "
		f"cells changed: {stats.cells_changed}"
	)


if __name__ == "__main__":
	main()
