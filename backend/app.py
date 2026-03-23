import os
import sys
import traceback
from pathlib import Path

# Add src to the path so modules load correctly
src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from flask import Flask, request, jsonify, render_template
import pandas as pd
from Classification.pipeline import SupportCascadePipeline

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
pipeline_instance = None


def _find_latest_manifest() -> Path:
    """Return the most recently modified training_manifest.json under backend/."""
    backend_dir = Path(__file__).resolve().parent
    candidates = sorted(
        backend_dir.rglob("training_manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No training_manifest.json found under backend/")
    return candidates[0]


def get_pipeline():
    global pipeline_instance
    if pipeline_instance is None:
        manifest_path = _find_latest_manifest()
        print(f"Loading SupportCascadePipeline from {manifest_path}...", flush=True)
        pipeline_instance = SupportCascadePipeline.from_manifest(str(manifest_path))
        print("Pipeline loaded successfully.", flush=True)
    return pipeline_instance


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_text", methods=["POST"])
def predict_text():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = str(data["text"]).strip()
        if not text:
            return jsonify({"error": "Empty ticket text"}), 400

        pipeline = get_pipeline()
        ticket = {
            "Ticket ID": "adhoc_text",
            "Ticket Subject": "",
            "Ticket Description": text,
        }
        prediction = pipeline.predict_ticket_record(ticket=ticket, ticket_id="adhoc_text")

        return jsonify({
            "success": True,
            "prediction": prediction.model_dump(),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not file.filename.endswith(".csv"):
            return jsonify({"error": "Only CSV files are supported"}), 400

        df = pd.read_csv(file)

        # Ensure we have the required column
        if "Ticket Description" not in df.columns:
            df["Ticket Description"] = df.iloc[:, 0].astype(str)
        if "Ticket Subject" not in df.columns:
            df["Ticket Subject"] = ""

        pipeline = get_pipeline()

        results = []
        for idx, row in df.iterrows():
            ticket = row.to_dict()
            ticket_id = str(ticket.get("ticket_id") or ticket.get("Ticket ID") or idx)
            try:
                record = pipeline.predict_ticket_record(ticket=ticket, ticket_id=ticket_id)
                results.append(record.model_dump())
            except Exception as exc:
                results.append({
                    "ticket_id": ticket_id,
                    "predicted_category": "",
                    "predicted_priority": "",
                    "top_3_retrieved_sources": "",
                    "draft_response": f"Error: {exc}",
                    "confidence_score": 0.0,
                    "abstain_flag": True,
                    "citations": "",
                    "tier_used": "error",
                    "routing_action": "manual_review",
                    "retrieval_max_score": 0.0,
                })

        return jsonify({
            "success": True,
            "results": results,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting server... Access it at http://127.0.0.1:{port}", flush=True)
    get_pipeline()
    app.run(host="0.0.0.0", port=port, debug=False)
