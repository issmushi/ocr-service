# RussianDocsOCR provenance

The embedded passport engine is derived from
[RussianDocsOCR 4.8.1](https://github.com/protei300/RussianDocsOCR/tree/65c88400d5d76289d754bdcbf427eeed374ac169),
commit `65c88400d5d76289d754bdcbf427eeed374ac169` (2026-10-18).
Its MIT license and upstream scope/third-party notice are retained verbatim as
`LICENSE.upstream` and `NOTICE.upstream`. Upstream states that the MIT license
covers its source and trained weights. The fonts and language ports mentioned
in that notice are not included here.

The integration keeps the document, orientation, text-field, word, accurate
Cyrillic/Latin OCR, and registration-address paths used by the service.
Quality, anti-spoofing, and unsupported CoreML/OpenVINO backends are excluded. Template
page registration and its image assets are not included; enabling it is rejected.
AddressLines, AddressTextKind, and RegistrationStamps load lazily when a
registration page is read. The address-reading and stamp-association stages
are retained from upstream without changes.
Document geometry helpers and the latest passport parsing/word assembly remain.
The new DocDetect stage selects the largest document before classification,
preserving its source resolution. The stage-based upstream pipeline layout is
retained, with excluded stages removed and compatibility imports preserved.
`Pipeline.process_img` defaults to no quality checks; requesting excluded checks
raises an explicit error. The service keeps its own public HTTP contracts.

Local patches resolve model paths through `RUSSIAN_OCR_MODELS_DIR`, preload GPU
runtime libraries when requested, preserve CPU-only inference availability, and
retain upstream dtype-aware ONNX inputs, session warmup, and GPU run locking.
Accurate OCR uses CPU by default even when detector models use GPU. Only the
accurate OCR model tier is included. No legacy OCR or model-format aliases remain.

The ten required ONNX model sets follow upstream `models-v11`, pinned by path,
size, SHA-256, release tag, and asset name in `app/ocr_engine/russian/models.lock.json`.
Unchanged files are fetched from their original releases; DocDetect and TextFields
come from v10, and DocTypeAngles weights and centers come from v11.
RegistrationStamps comes from v11; AddressLines and AddressTextKind come from v4.
Model binaries are downloaded and verified by the service installer; they are
not stored in this repository. Upstream samples, tests, services, scripts, ports,
benchmark output, and Python bytecode are excluded.
