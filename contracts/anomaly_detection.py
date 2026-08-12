# {
#   "Seq": [
#     { "Depends": "py-lib-genlayer-embeddings:0bmbm3cyfwxsyh454z53vxqjf47wz2q7smcqp1q4g4a6k2kidnyk" },
#     { "Depends": "py-genlayer:1jb45aa8ynh2a9c9xn3b7qqh8sm5q93hwfp7jqmwsfhh8jpz09h6" }
#   ]
# }

"""
AnomalyDetection — a vector-semantic change tracker with LLM adjudication.

Indexes free-text observations as embeddings and classifies an incoming
observation as *novel* (a genuinely new anomaly) or a *near-duplicate* of an
existing one. Embedding + nearest-neighbor search shortlists the closest
candidate deterministically, then an LLM makes the actual novelty decision
based on that evidence. Because the LLM call is non-deterministic, it runs
inside a consensus block (`gl.vm.run_nondet`) whose validator independently
binds the consequential output: only a verdict that passes validation drives
the stored record, the `is_novel` flag, and the returned reason.

"""

import numpy as np
import typing
from dataclasses import dataclass

from genlayer import *
import genlayer_embeddings as gle


@allow_storage
@dataclass
class Observation:
    text: str
    log_id: u256
    source: str
    is_novel: bool = True
    reason: str = ""


class AnomalyDetection(gl.Contract):
    vector_store: gle.VecDB[
        np.float32, typing.Literal[384], Observation, gle.EuclideanDistanceSquared
    ]
    next_id: u256
    novel_threshold: float

    def __init__(self, novel_threshold: int = 80):
        self.next_id = u256(0)
        self.novel_threshold = float(novel_threshold) / 100.0

    def get_embedding_generator(self):
        return gle.SentenceTransformer("all-MiniLM-L6-v2")

    def get_embedding(
        self, text: str
    ) -> np.ndarray[tuple[typing.Literal[384]], np.dtypes.Float32DType]:
        return self.get_embedding_generator()(text)

    def _similarity(self, distance: float) -> float:
        return 1.0 / (1.0 + float(distance))

    def _classify(self, verdict: str, fallback: bool) -> tuple[bool, str]:
        text = verdict.strip() if verdict else ""
        first = text.split(maxsplit=1)[0].upper() if text else ""
        if first == "DUPLICATE":
            return False, text
        if first == "NOVEL":
            return True, text
        return fallback, text

    @gl.public.view
    def get_closest(self, text: str) -> dict | None:
        emb = self.get_embedding(text)
        results = list(self.vector_store.knn(emb, 1))
        if len(results) == 0:
            return None
        r = results[0]
        return {
            "text": r.value.text,
            "source": r.value.source,
            "log_id": str(r.value.log_id),
            "similarity": "{:.4f}".format(self._similarity(r.distance)),
        }

    @gl.public.view
    def is_novel(self, text: str) -> bool:
        emb = self.get_embedding(text)
        results = list(self.vector_store.knn(emb, 1))
        if len(results) == 0:
            return True
        return bool(self._similarity(results[0].distance) < self.novel_threshold)

    @gl.public.view
    def observation_count(self) -> int:
        return len(self.vector_store)

    @gl.public.write
    def add_observation(self, log: str, source: str = "") -> dict:
        emb = self.get_embedding(log)
        results = list(self.vector_store.knn(emb, 1))
        threshold = float(self.novel_threshold)

        if len(results) == 0:
            novel = True
            reason = "No prior observations to compare against."
        else:
            nearest = results[0]
            neighbor_text = nearest.value.text
            similarity = self._similarity(nearest.distance)
            threshold_novel = bool(similarity < threshold)

            def classification_task() -> str:
                prompt = (
                    "You adjudicate anomaly logs. Decide whether the incoming log "
                    "is a genuinely NEW anomaly (NOVEL) or a near-duplicate of an "
                    "already recorded one (DUPLICATE).\n\n"
                    f"Incoming log: {log}\n"
                    f"Closest stored observation: {neighbor_text}\n"
                    f"Embedding similarity to closest: {similarity:.4f} "
                    f"(novelty threshold: {threshold:.2f})\n"
                    "Similarity below the threshold suggests NOVEL; at or above it "
                    "suggests DUPLICATE. Use it as evidence, not a rule: paraphrases "
                    "of the same incident are DUPLICATE, a different incident is NOVEL.\n\n"
                    "Return exactly one line: NOVEL <short reason> or DUPLICATE <short reason>."
                )
                return gl.nondet.exec_prompt(prompt)

            def verdict_validator(result) -> bool:
                if not isinstance(result, gl.vm.Return):
                    return False
                raw = str(result.calldata).strip()
                first = raw.split(maxsplit=1)[0].upper() if raw else ""
                return first in ("NOVEL", "DUPLICATE")

            verdict = gl.vm.run_nondet(classification_task, verdict_validator)
            novel, reason = self._classify(str(verdict), threshold_novel)

        obs = Observation(
            text=log,
            log_id=self.next_id,
            source=source,
            is_novel=novel,
            reason=reason,
        )
        self.next_id = u256(int(self.next_id) + 1)
        self.vector_store.insert(emb, obs)

        return {
            "log_id": str(obs.log_id),
            "is_novel": novel,
            "reason": reason,
            "count": len(self.vector_store),
        }

    @gl.public.write
    def remove_observation(self, log_id: int) -> dict:
        removed = False
        for el in self.vector_store:
            if int(el.value.log_id) == int(log_id):
                el.remove()
                removed = True
        return {"removed": removed, "count": len(self.vector_store)}