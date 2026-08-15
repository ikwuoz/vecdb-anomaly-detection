# GenLayer Anomaly Detection

An intelligent contract on [GenLayer](https://genlayer.com/) that indexes free-text observations as embeddings and classifies each incoming observation as **novel** (a genuinely new anomaly) or a **near-duplicate** of an existing one. It uses the `genlayer_embeddings` SDK (`VecDB` + `SentenceTransformer`) running inside the GenVM, and an LLM that adjudicates each novelty decision inside a consensus block (`gl.vm.run_nondet`), so GenLayer validators independently bind the stored verdict.

This is a generic vector-semantic primitive — useful for duplicate detection, outage gating, and intent routing.

## What's included
- `contracts/anomaly_detection.py` — the `AnomalyDetection` contract
- **Direct mode tests** — fast, in-memory unit tests with a deterministic embedding mock (~ms per test)
- **Contract linting** — static analysis via the GenVM linter
- **Deployment + demo scripts** — deploy to GenLayer Studio and drive the full lifecycle
- **CI pipeline** — GitHub Actions for linting and direct tests

## Requirements
- Python >= 3.12
- [GenLayer CLI](https://github.com/genlayerlabs/genlayer-cli) globally installed: `npm install -g genlayer`
- GenLayer Studio (hosted): [studio.genlayer.com](https://studio.genlayer.com/)

## Project Structure
```
contracts/                    # Python intelligent contracts
  anomaly_detection.py        #   the anomaly detection contract
tests/
  direct/                     # Fast in-memory tests (no Studio required)
    test_anomaly_detection.py
deploy/                       # TypeScript deployment scripts
scripts/
  anomaly_demo.mjs            # Lifecycle demo against GenLayer Studio
gltest.config.yaml            # Test runner network configuration
pyproject.toml                # Python/pytest configuration
.github/workflows/            # CI pipeline
```

## Contract API
| Method | Type | Description |
|--------|------|-------------|
| `AnomalyDetection(novel_threshold=80)` | ctor | Deploy with a novelty threshold; `80` means 0.80 (integer percentage, 0–100, since floats aren't calldata-encodable) |
| `add_observation(log, source)` | write | Embed an observation, have the LLM adjudicate novelty (consensus-validated), and store it; returns `{ log_id, is_novel, reason, count }` |
| `remove_observation(log_id)` | write | Remove an observation by id; returns `{ removed, count }` |
| `is_novel(text)` | view | Fast deterministic check: True if `text` is dissimilar to everything stored |
| `get_closest(text)` | view | The stored observation nearest to `text`, with similarity |
| `observation_count()` | view | Number of stored observations |

Novelty is decided by an LLM adjudication: embedding + nearest-neighbor search shortlists the closest candidate deterministically, then the LLM classifies the incoming log as **NOVEL** or **DUPLICATE** given that evidence. Because the LLM call is non-deterministic, it runs inside `gl.vm.run_nondet`, whose validator **independently recomputes** the novelty decision (same observation, nearest match, similarity, threshold) and only agrees when the leader's verdict token matches it — so an opposite or malformed classification fails consensus and cannot change the stored result. The similarity threshold is used as the rule in the prompt and as the leader-side fallback.

## Quick Start

### 1. Set up Python environment
```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Lint the contract
```shell
genvm-lint check contracts/anomaly_detection.py
```

### 3. Run direct mode tests
Direct mode runs the contract in-memory with a deterministic embedding mock — no Studio required:
```shell
pytest tests/direct/ -v
```

### 4. Run the lifecycle demo (GenLayer Studio)
The demo deploys the contract (or reuses `ANOMALY_CONTRACT_ADDRESS`), then drives the full lifecycle with semantic assertions:
```shell
npm install
node scripts/anomaly_demo.mjs
```

To reuse a deployed contract:
```shell
ANOMALY_CONTRACT_ADDRESS=0x... node scripts/anomaly_demo.mjs
```

## Testing Strategy
| Test Type | Command | Speed | Requires Studio |
|-----------|---------|-------|-----------------|
| **Lint** | `genvm-lint check contracts/anomaly_detection.py` | ~250ms | No |
| **Direct** | `pytest tests/direct/ -v` | ~ms/test | No |
| **Demo** | `node scripts/anomaly_demo.mjs` | ~min/run | Yes |

## Documentation
For detailed information, see the [GenLayer documentation](https://docs.genlayer.com/).

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
