"""
Serve CEFR model as a FastAPI endpoint on Modal.

Usage:
    uv run modal deploy serve.py
    # Then: curl https://your-app.modal.run/score -X POST -H "Content-Type: application/json" -d '{"text": "..."}'
"""
import modal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = modal.App("cefr-api")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.6.0",
        "transformers>=5.14.1",
        "fastapi>=0.104.0",
        "sentencepiece>=0.1.99",
    )
    .add_local_file("model.py", "/app/model.py")
)

volume = modal.Volume.from_name("cefr-models")

# FastAPI app
web_app = FastAPI(title="CEFR Scoring API")


class ScoreRequest(BaseModel):
    text: str


class ScoreResponse(BaseModel):
    score: float
    cefr_level: str
    confidence: str


@app.cls(
    image=image,
    gpu="T4",
    volumes={"/vol": volume},
    scaledown_window=60,  # Keep warm for 1 minute
)
class CEFRService:
    """CEFR scoring service with model lifecycle management."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None

    @modal.enter()
    def startup(self):
        """Load model on container startup."""
        import torch
        from transformers import AutoTokenizer
        import sys
        sys.path.insert(0, "/app")
        from model import CEFRModel

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model on {self.device}...")
        self.model = CEFRModel().float()
        self.model.load_state_dict(
            torch.load("/vol/best_model.pt", map_location=self.device, weights_only=True)
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained("/vol/tokenizer")
        print("Model loaded!")

    @modal.asgi_app()
    def serve(self):
        """Return the FastAPI app."""

        @web_app.get("/health")
        def health():
            return {"status": "healthy", "model_loaded": self.model is not None}

        @web_app.post("/score", response_model=ScoreResponse)
        def score_essay(request: ScoreRequest):
            """Score an essay and return CEFR level."""
            import torch
            import sys
            sys.path.insert(0, "/app")
            from model import score_to_cefr

            if self.model is None:
                raise HTTPException(500, "Model not loaded")

            if len(request.text.strip()) < 10:
                raise HTTPException(400, "Text too short (min 10 characters)")

            # Tokenize
            encoding = self.tokenizer(
                request.text,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )

            # Move inputs to same device as model
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            # Predict
            with torch.no_grad():
                score = self.model(input_ids, attention_mask).item()

            # Clamp to valid range
            score = max(1.0, min(6.0, score))
            cefr = score_to_cefr(score)

            # Confidence heuristic based on distance to nearest CEFR boundary.
            # Scores near boundaries (e.g., 2.48 between A2/B1) are less certain.
            boundaries = {1.5, 2.5, 3.5, 4.5, 5.5}
            min_dist = min(abs(score - b) for b in boundaries)
            confidence = "high" if min_dist > 0.3 else "medium" if min_dist > 0.15 else "low"

            return ScoreResponse(
                score=round(score, 2),
                cefr_level=cefr,
                confidence=confidence,
            )

        return web_app
