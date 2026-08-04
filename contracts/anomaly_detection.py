# {
#   "Seq": [
#     { "Depends": "py-genlayer:1jb45aa8ynh2a9c9xn3b7qqh8sm5q93hwfp7jqmwsfhh8jpz09h6" },
#     { "Depends": "py-lib-genlayer-embeddings:0bmbm3cyfwxsyh454z53vxqjf47wz2q7smcqp1q4g4a6k2kidnyk" }
#   ]
# }

"""
AnomalyDetection — a reusable vector-semantic change tracker.

Indexes free-text observations as embeddings and classifies an incoming
observation as *novel* (dissimilar to everything already stored) or a
*near-duplicate* of an existing one, using a configurable similarity
threshold. The threshold core is the same primitive that powers duplicate
detection, outage gating, and intent routing, so it is kept generic.

Confirmed genlayer_embeddings SDK surface:
  import genlayer_embeddings as gle
  gle.VecDB[np.float32, typing.Literal[384], V]
      .insert(embedding, value) -> Id ; .knn(embedding, k) -> iterator
      elements expose .value, .distance, .id ; VecDB supports len()
  gle.SentenceTransformer("all-MiniLM-L6-v2")(text) -> (384,) float32
  Distance is EuclideanDistanceSquared; similarity ~= 1/(1+distance).
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


class AnomalyDetection(gl.Contract):
    vector_store: gle.VecDB[
        np.float32, typing.Literal[384], Observation, gle.EuclideanDistanceSquared
    ]
    next_id: u256
    novel_threshold: float

    def __init__(self):
        self.next_id = u256(0)
        self.novel_threshold = float(0.80)

    def get_embedding_generator(self):
        return gle.SentenceTransformer("all-MiniLM-L6-v2")

    def get_embedding(
        self, text: str
    ) -> np.ndarray[tuple[typing.Literal[384]], np.dtypes.Float32DType]:
        return self.get_embedding_generator()(text)

    def _similarity(self, distance: float) -> float:
        return 1.0 / (1.0 + float(distance))

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
            "similarity": self._similarity(r.distance),
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
        novel = len(results) == 0 or bool(
            self._similarity(results[0].distance) < self.novel_threshold
        )

        obs = Observation(text=log, log_id=self.next_id, source=source)
        self.next_id = u256(int(self.next_id) + 1)
        self.vector_store.insert(emb, obs)

        return {
            "log_id": str(obs.log_id),
            "is_novel": novel,
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