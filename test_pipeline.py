import sys, io
# Force UTF-8 output on Windows to avoid CP1252 UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import sys
from pathlib import Path

# Add src to sys.path
repo_root = Path(__file__).resolve().parent
src_dir = repo_root / "backend" / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from Classification.pipeline import SupportCascadePipeline
from Classification.tier_1 import normalize_ticket_text


def _format_classifier_info(tier_used: str) -> tuple[str, str]:
    tier = str(tier_used or "")
    if tier.startswith("groq:"):
        model_name = tier.split(":", 1)[1]
        return "Tier 3 (Groq)", model_name
    if tier == "groq":
        return "Tier 3 (Groq)", "unknown"
    if tier == "setfit":
        return "Tier 2 (SetFit)", "n/a"
    if tier == "logreg":
        return "Tier 1 (LogReg)", "n/a"
    return f"Unknown ({tier or 'n/a'})", "n/a"

def run_test_with_scores(pipeline, name, ticket, fast_mode: bool):
    print(f"\n──────────────────────────────")
    print(f"TEST: {name}")
    print(f"  Subject : {ticket['Ticket Subject']}")
    
    # Let's see the step-by-step scores
    # Phase 1: Tier 1 (Tfidf)
    tier1 = pipeline.tier1_router if pipeline.tier1_router else pipeline.tier1_baseline
    t1 = tier1.predict(ticket['Ticket Subject'])
    print(f"  Tier 1 Conf: {t1.confidence_score:.2f} (Needs 0.90 to stop)")
    
    # Phase 2: Tier 2 (SetFit)
    if t1.abstain_flag:
        t2 = pipeline.tier2_classifier.predict(ticket['Ticket Subject'] + " " + ticket['Ticket Description'])
        print(f"  Tier 2 Conf: {t2.confidence_score:.2f} (Needs 0.45 to stop)")
    else:
        print(f"  Tier 1 won!")

    # Final Result
    try:
        if fast_mode:
            text = normalize_ticket_text(
                f"{ticket.get('Ticket Subject', '')} {ticket.get('Ticket Description', '')}".strip()
            )
            res = pipeline._predict_classification(text)
        else:
            res = pipeline.predict_ticket(ticket)
        classifier, groq_model = _format_classifier_info(str(res.tier_used or ""))
        final_conf = float(res.confidence_score or 0.0)
        print(f"  FINAL CONF: {final_conf:.2f}")
        print(f"  CLASSIFIER: {classifier}")
        if classifier.startswith("Tier 3 (Groq)"):
            print(f"  GROQ MODEL: {groq_model}")
        print(f"  FINAL TIER: {str(res.tier_used or '').upper()}")
        print(f"  PREDICTION: {res.predicted_category} ({res.predicted_priority})")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")

parser = argparse.ArgumentParser(description="Run quick classification tests for the support cascade")
parser.add_argument(
    "--full",
    action="store_true",
    help="Run full pipeline including RAG generation (slower).",
)
args = parser.parse_args()
fast_mode = not args.full

# 1. Load pipeline
manifest_file = repo_root / "backend" / "results" / "training_manifest.json"
print("==============================")
print(f"Loading SupportCascadePipeline ...")
print(f"Manifest: {manifest_file}")
print(f"Mode: {'FAST (classification only)' if fast_mode else 'FULL (classification + RAG)'}")
print("==============================")

try:
    pipeline = SupportCascadePipeline.from_manifest(
        str(manifest_file),
        enable_tier3_llm=not fast_mode,
    )
    print("✓ Pipeline loaded successfully")
except Exception as e:
    print(f"✗ ERROR Loading Pipeline: {e}")
    sys.exit(1)

# Example 1: Clear Billing (Should ideally be Tier 1 or 2)
ticket1 = {
    "Ticket Subject": "Billing charge error on my invoice",
    "Ticket Description": "I was charged twice for my subscription this month. Please refund the $25 extra charge.",
    "Customer Name": "Alice Smith",
    "Customer Email": "alice@example.com"
}
run_test_with_scores(pipeline, "Tier-1 target  – clear billing issue", ticket1, fast_mode)

# Example 2: Technical/Reset (Should be Tier 2)
ticket2 = {
    "Ticket Subject": "Unable to reset password",
    "Ticket Description": "I cannot receive reset email since morning. I have checked my spam folder and whitelist but nothing arrived.",
    "Customer Name": "Bob Jones",
    "Customer Email": "bob@example.com"
}
run_test_with_scores(pipeline, "Tier-2 target  – password reset issue", ticket2, fast_mode)

# Example 3: Ambiguous (Should be Tier 3)
ticket3 = {
    "Ticket Subject": "Something is just not right",
    "Ticket Description": "Everything was fine until yesterday but now it feels off. Can someone check my account?",
    "Customer Name": "Charlie Brown",
    "Customer Email": "charlie@example.com"
}
run_test_with_scores(pipeline, "Tier-3 target  – vague/ambiguous issue", ticket3, fast_mode)

print("\n==============================")
print("Done.")