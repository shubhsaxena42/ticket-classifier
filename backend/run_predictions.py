"""Generate predictions.csv from an input CSV of support tickets.

Usage:
    python run_predictions.py --input eval_tickets.csv --output predictions.csv
    python run_predictions.py --input eval_tickets.csv  # defaults to predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path so Classification/RAG/generation imports resolve.
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from Classification.pipeline import SupportCascadePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full support cascade pipeline on an eval CSV.")
    parser.add_argument("--input", required=True, help="Path to input CSV with ticket data.")
    parser.add_argument("--output", default="predictions.csv", help="Path for output predictions CSV (default: predictions.csv).")
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).resolve().parent / "results" / "training_manifest.json"),
        help="Path to training_manifest.json.",
    )
    parser.add_argument("--no-tier3", action="store_true", help="Disable Tier 3 LLM calls.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    print(f"Loading pipeline from {args.manifest} ...")
    pipeline = SupportCascadePipeline.from_manifest(
        manifest_path=args.manifest,
        enable_tier3_llm=not args.no_tier3,
    )
    print("Pipeline loaded.\n")

    df = pd.read_csv(input_path)
    print(f"Input: {len(df)} tickets from {input_path.name}")

    rows = []
    for idx, row in df.iterrows():
        ticket = row.to_dict()
        ticket_id = str(ticket.get("ticket_id") or ticket.get("Ticket ID") or idx)
        print(f"  [{idx + 1}/{len(df)}] Processing ticket {ticket_id} ...", end=" ", flush=True)
        try:
            record = pipeline.predict_ticket_record(ticket=ticket, ticket_id=ticket_id)
            rows.append(record.model_dump())
            print(f"OK  ({record.tier_used}, {record.routing_action})")
        except Exception as exc:
            print(f"FAILED: {exc}")
            rows.append({
                "ticket_id": ticket_id,
                "predicted_category": "",
                "predicted_priority": "",
                "top_3_retrieved_sources": "",
                "draft_response": "",
                "confidence_score": 0.0,
                "abstain_flag": True,
                "citations": "",
                "tier_used": "error",
                "routing_action": "manual_review",
                "retrieval_max_score": 0.0,
            })

    out_df = pd.DataFrame(rows)
    output_path = Path(args.output).resolve()
    out_df.to_csv(output_path, index=False)
    print(f"\nWrote {len(out_df)} predictions to {output_path}")


if __name__ == "__main__":
    main()
