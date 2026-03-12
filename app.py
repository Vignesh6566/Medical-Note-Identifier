import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# ── Model loading ──────────────────────────────────────────────────────────────
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
print(f"[INFO] Loading tokenizer and model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()
print("[INFO] Model loaded successfully.")

# ── Label definitions with rich clinical seed phrases ─────────────────────────
LABELS = {
    "Cardiology": [
        "chest pain heart attack myocardial infarction coronary artery disease arrhythmia",
        "palpitations angina shortness of breath left arm pain cardiac catheterization",
        "atrial fibrillation hypertension blood pressure ECG echocardiogram",
    ],
    "Respiratory": [
        "cough breathing difficulty asthma bronchitis pneumonia lung",
        "wheezing shortness of breath oxygen saturation pulmonary COPD",
        "sputum respiratory infection spirometry inhaler nebulizer",
    ],
    "Endocrine": [
        "diabetes insulin glucose blood sugar thyroid hormone",
        "hyperglycemia hypoglycemia HbA1c endocrine metabolism",
        "hypothyroidism hyperthyroidism adrenal cortisol pancreas",
    ],
    "Neurology": [
        "headache migraine seizure stroke brain nerve",
        "dizziness tremor Parkinson multiple sclerosis epilepsy",
        "MRI brain scan neurological cognitive memory loss",
    ],
    "Infectious Disease": [
        "fever infection bacteria virus antibiotic sepsis",
        "COVID influenza pneumonia cellulitis urinary tract infection",
        "culture sensitivity contagious immunocompromised pathogen",
    ],
    "Routine Checkup": [
        "annual physical wellness exam preventive screening",
        "routine blood work cholesterol vaccination immunization",
        "general health checkup no acute complaints vital signs normal",
    ],
}


def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling over token embeddings, respecting attention mask."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)


def get_embedding(text: str) -> torch.Tensor:
    """Tokenize text and return its mean-pooled CLS embedding."""
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
    return mean_pool(outputs.last_hidden_state, encoded["attention_mask"])


# Pre-compute label embeddings (average over seed phrases per label)
print("[INFO] Pre-computing label embeddings...")
label_names = list(LABELS.keys())
label_embeddings = []
for label, phrases in LABELS.items():
    phrase_embs = torch.cat([get_embedding(p) for p in phrases], dim=0)
    label_embeddings.append(phrase_embs.mean(dim=0, keepdim=True))

label_embeddings = torch.cat(label_embeddings, dim=0)           # (num_labels, hidden)
label_embeddings = F.normalize(label_embeddings, p=2, dim=-1)   # unit-normalise
print("[INFO] Label embeddings ready.")


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Embed input
    query_emb = get_embedding(text)
    query_emb = F.normalize(query_emb, p=2, dim=-1)  # (1, hidden)

    # Cosine similarities → softmax probabilities
    sims = (query_emb @ label_embeddings.T).squeeze(0)  # (num_labels,)
    probs = F.softmax(sims * 10.0, dim=0)               # temperature scaling

    best_idx = int(torch.argmax(probs).item())
    prediction = label_names[best_idx]
    confidence = float(probs[best_idx].item())

    all_scores = {label_names[i]: round(float(probs[i].item()), 4) for i in range(len(label_names))}

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "all_scores": all_scores,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
