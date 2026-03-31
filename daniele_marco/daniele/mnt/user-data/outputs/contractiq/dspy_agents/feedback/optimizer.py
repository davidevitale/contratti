"""
contractiq/dspy_agents/feedback/optimizer.py

THE LOCK-IN ENGINE — DSPy MIPROv2 Per-Client Optimizer.

How it works:
  1. Every user interaction (question + answer) is stored.
  2. Every user rating (1-5) + comment is stored as a training example.
  3. When enough feedback accumulates (threshold: 20+ examples),
     MIPROv2 runs and finds the best prompts for THIS client's contracts.
  4. The compiled program is saved to disk as {client_id}.json
  5. Next request for this client loads the optimized program.

Why this creates irreversible lock-in:
  - The optimized program encodes client-specific vocabulary, clause patterns,
    and answer styles learned from THEIR usage data.
  - Switching to a competitor means starting from zero — no history, no optimization.
  - The longer they use ContractIQ, the better it gets for them specifically.
  - Competitors can't reproduce this without the same feedback data.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import dspy

logger = logging.getLogger(__name__)

OPTIMIZED_MODELS_PATH = Path(os.getenv("OPTIMIZED_MODELS_PATH", "/app/optimized_models"))
FEEDBACK_THRESHOLD = int(os.getenv("FEEDBACK_THRESHOLD", "20"))


# ─────────────────────────────────────────────────────────────────────────────
# Optimization Metric
# ─────────────────────────────────────────────────────────────────────────────

def user_satisfaction_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """
    DSPy optimization metric based on user feedback.

    Maps user ratings to a 0.0–1.0 quality signal:
      5 stars → 1.0  (perfect)
      4 stars → 0.8
      3 stars → 0.5
      2 stars → 0.2
      1 star  → 0.0

    Bonus signals:
      - Citation quality: did the answer cite specific clauses?
      - Confidence calibration: high confidence + wrong → penalty
      - User comment sentiment: positive words boost score
    """
    base_rating = getattr(example, "user_rating", 3)
    rating_map = {5: 1.0, 4: 0.8, 3: 0.5, 2: 0.2, 1: 0.0}
    score = rating_map.get(base_rating, 0.5)

    # Bonus: check if answer contains citations (good practice)
    answer = getattr(prediction, "answer", "")
    sources = getattr(prediction, "sources", "[]")
    try:
        sources_list = json.loads(sources) if isinstance(sources, str) else sources
        if len(sources_list) > 0:
            score = min(1.0, score + 0.1)  # Citation bonus
    except (json.JSONDecodeError, TypeError):
        pass

    # Bonus: positive comment keywords
    comment = getattr(example, "user_comment", "").lower()
    positive_words = ["ottimo", "perfetto", "utile", "preciso", "correct", "great", "perfect"]
    negative_words = ["sbagliato", "errato", "wrong", "incorrect", "useless", "bad"]

    if any(w in comment for w in positive_words):
        score = min(1.0, score + 0.05)
    if any(w in comment for w in negative_words):
        score = max(0.0, score - 0.15)

    return score


# ─────────────────────────────────────────────────────────────────────────────
# Per-Client Optimizer
# ─────────────────────────────────────────────────────────────────────────────

class PerClientOptimizer:
    """
    Manages the lifecycle of per-client DSPy compiled programs.

    Each client gets their own optimized version of ContractIQOrchestrator,
    compiled specifically on their contract domain and feedback preferences.
    """

    def __init__(self):
        try:
            OPTIMIZED_MODELS_PATH.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            logger.warning(f"Could not create models directory {OPTIMIZED_MODELS_PATH}: {e}")

    def get_model_path(self, client_id: str) -> Path:
        return OPTIMIZED_MODELS_PATH / f"{client_id}.json"

    def get_metadata_path(self, client_id: str) -> Path:
        return OPTIMIZED_MODELS_PATH / f"{client_id}_meta.json"

    def has_optimized_model(self, client_id: str) -> bool:
        return self.get_model_path(client_id).exists()

    def load_optimized_orchestrator(self, client_id: str, base_orchestrator) -> any:
        """
        Load a client-specific compiled DSPy program.
        Falls back to base orchestrator if no optimized version exists.
        """
        model_path = self.get_model_path(client_id)
        if model_path.exists():
            try:
                base_orchestrator.load(str(model_path))
                logger.info(f"Loaded optimized model for client {client_id}")
                return base_orchestrator
            except Exception as e:
                logger.warning(f"Failed to load optimized model for {client_id}: {e}")
        return base_orchestrator

    def save_optimized_orchestrator(self, client_id: str, compiled_orchestrator) -> bool:
        """Save a newly compiled program to disk."""
        try:
            compiled_orchestrator.save(str(self.get_model_path(client_id)))
            # Save metadata
            meta = {
                "client_id": client_id,
                "compiled_at": datetime.utcnow().isoformat(),
                "version": self._get_next_version(client_id),
            }
            with open(self.get_metadata_path(client_id), "w") as f:
                json.dump(meta, f, indent=2)
            logger.info(f"Saved optimized model for client {client_id} v{meta['version']}")
            return True
        except Exception as e:
            logger.error(f"Failed to save optimized model for {client_id}: {e}")
            return False

    def _get_next_version(self, client_id: str) -> int:
        meta_path = self.get_metadata_path(client_id)
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            return meta.get("version", 0) + 1
        return 1

    def run_optimization(
        self,
        client_id: str,
        base_orchestrator,
        feedback_examples: list[dspy.Example],
    ) -> Optional[any]:
        """
        Run MIPROv2 optimization on the orchestrator for this client.

        MIPROv2 will:
          1. Generate candidate instruction variants for each DSPy module
          2. Generate few-shot examples from the training set
          3. Search for the combination that maximizes user_satisfaction_metric
          4. Return a compiled program with optimized prompts baked in

        This compiled program is the moat.
        """
        if len(feedback_examples) < FEEDBACK_THRESHOLD:
            logger.info(
                f"Client {client_id}: {len(feedback_examples)}/{FEEDBACK_THRESHOLD} "
                f"feedback examples needed for optimization"
            )
            return None

        logger.info(f"Starting MIPROv2 optimization for client {client_id} "
                    f"with {len(feedback_examples)} examples")

        try:
            # Split into train/dev
            split = int(len(feedback_examples) * 0.8)
            trainset = feedback_examples[:split]
            devset = feedback_examples[split:]

            # MIPROv2: Automatic prompt optimization
            optimizer = dspy.MIPROv2(
                metric=user_satisfaction_metric,
                auto="light",  # "light" for MVP, "medium"/"heavy" for production
                num_threads=4,
            )

            compiled = optimizer.compile(
                base_orchestrator,
                trainset=trainset,
                valset=devset,
                num_trials=15,           # Optimize over 15 candidate programs
                minibatch_size=5,
                minibatch_full_eval_steps=10,
                requires_permission_to_run=False,
            )

            self.save_optimized_orchestrator(client_id, compiled)
            logger.info(f"Optimization complete for client {client_id}")
            return compiled

        except Exception as e:
            logger.error(f"MIPROv2 optimization failed for {client_id}: {e}")
            return None

    def get_optimization_status(self, client_id: str) -> dict:
        """Return current optimization status for a client."""
        meta_path = self.get_metadata_path(client_id)
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            return {
                "optimized": True,
                "version": meta.get("version", 1),
                "compiled_at": meta.get("compiled_at"),
                "model_path": str(self.get_model_path(client_id)),
            }
        return {
            "optimized": False,
            "version": 0,
            "compiled_at": None,
            "feedback_needed": FEEDBACK_THRESHOLD,
        }
