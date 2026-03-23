"""Run end-to-end inference on an eval CSV and write predictions.csv.

This script loads SupportCascadePipeline and produces assignment-style outputs
for each ticket row, including grounded retrieval/generation fields.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from Classification.pipeline import SupportCascadePipeline  # type: ignore[import-not-found]


def _resolve_ticket_fields(row: Dict[str, object]) -> Tuple[str, str, str]:
    """Resolve ticket id, subject, and message from flexible CSV column names.

    Args:
        row: One input CSV row as a dictionary.

    Returns:
        Tuple of (ticket_id, subject, message).
    """

    ticket_id = str(row.get("ticket_id") or row.get("Ticket ID") or row.get("id") or "")
    subject = str(row.get("subject") or row.get("Ticket Subject") or "")
    message = str(row.get("message") or row.get("Ticket Description") or "")
    return ticket_id, subject, message


def run_eval(manifest_path: Path, eval_csv_path: Path, output_csv_path: Path) -> Path:
    """Execute full inference over an eval set and write predictions.csv.

    Args:
        manifest_path: Path to training/inference manifest JSON.
        eval_csv_path: Path to tickets_eval CSV.
        output_csv_path: Output path for predictions CSV.

    Returns:
        Path to the written predictions CSV file.
    """

    pipeline = SupportCascadePipeline.from_manifest(
        str(manifest_path),
        enable_tier3_llm=False,
    )
    eval_df = pd.read_csv(eval_csv_path)

    rows = []
    for idx, row in eval_df.iterrows():
        row_dict = row.to_dict()
        ticket_id, subject, message = _resolve_ticket_fields(row_dict)
        if not ticket_id:
            ticket_id = f"row_{idx + 1}"

        ticket = {
            "Ticket ID": ticket_id,
            "Ticket Subject": subject,
            "Ticket Description": message,
        }

        prediction = pipeline.predict_ticket_record(ticket=ticket, ticket_id=ticket_id)
        rows.append(prediction.model_dump())

    out_df = pd.DataFrame(rows)
    out_df = out_df[
        [
            "ticket_id",
            "predicted_category",
            "predicted_priority",
            "top_3_retrieved_sources",
            "draft_response",
            "confidence_score",
            "abstain_flag",
            "citations",
            "tier_used",
            "routing_action",
            "retrieval_max_score",
        ]
    ]
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv_path, index=False)
    return output_csv_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run eval inference and write predictions.csv")
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).resolve().parent / "results" / "training_manifest.json"),
        help="Path to training manifest JSON.",
    )
    parser.add_argument(
        "--eval_csv",
        required=True,
        help="Path to evaluation CSV (tickets_eval.csv).",
    )
    parser.add_argument(
        "--output_csv",
        default=str(Path(__file__).resolve().parent / "results" / "predictions.csv"),
        help="Path for output predictions.csv.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    output = run_eval(
        manifest_path=Path(args.manifest),
        eval_csv_path=Path(args.eval_csv),
        output_csv_path=Path(args.output_csv),
    )
    print(f"Wrote predictions to: {output}")


if __name__ == "__main__":
    main()
