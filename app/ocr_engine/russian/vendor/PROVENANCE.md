# RussianDocsOCR provenance

The embedded passport engine is derived from
[RussianDocsOCR 4.6.1](https://github.com/protei300/RussianDocsOCR/tree/bd0028aa8e9d3a30d2de4df8b672185bfcd984ba),
commit `bd0028aa8e9d3a30d2de4df8b672185bfcd984ba` (2026-09-28).
Its MIT license and upstream scope/third-party notice are retained verbatim as
`LICENSE.upstream` and `NOTICE.upstream`. Upstream states that the MIT license
covers its source and trained weights. The fonts and language ports mentioned
in that notice are not included here.

The integration keeps the document, orientation, text-field, word, and accurate
Cyrillic/Latin OCR paths used by the service. Quality, anti-spoofing, address
recognition, and unsupported CoreML/OpenVINO backends are excluded. Template
page registration and its image assets are not included; enabling it is rejected.
Document geometry helpers and the latest passport parsing/word assembly remain.
`Pipeline.process_img` defaults to no quality checks; requesting excluded checks
raises an explicit error. The service keeps its own public HTTP contracts.

Local patches resolve model paths through `RUSSIAN_OCR_MODELS_DIR`, preload GPU
runtime libraries when requested, preserve CPU-only inference availability, and
retain upstream dtype-aware ONNX inputs, session warmup, and GPU run locking.
Accurate OCR uses CPU by default even when detector models use GPU. Only the
accurate OCR model tier is included. No legacy OCR or model-format aliases remain.

The six required ONNX model sets come from upstream `models-v8`, pinned by path,
size, SHA-256, and release asset name in `app/ocr_engine/russian/models.lock.json`.
Model binaries are downloaded and verified by the service installer; they are
not stored in this repository. Upstream samples, tests, services, scripts, ports,
benchmark output, and Python bytecode are excluded.
