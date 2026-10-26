# Catalogue-driven message engine: proof of concept

Synthetic-data proof of concept for IDEaS CH17 (RF data fusion for C2). Python 3.10+, `lxml` for XSD validation, `pytest` for tests.

```
python3 -m pip install lxml pytest
python3 -m pytest -q      # 148 tests
python3 demo.py           # codec fidelity, sizes, corruption     -> results/
python3 ddil_demo.py      # emulated links, 4 encodings, 20 seeds -> results/ddil_results.json
python3 extract_demo.py   # free-text extraction accuracy         -> results/extract_results.json
python3 fusion_demo.py    # multi-source fusion on 20 scenarios   -> results/fusion_results.json
```

## What it does

- **Catalogue as data** (`catalogue/demo_catalogue.json`): four synthetic message families (call for fire, MEDEVAC request, logistics status, air/surface contact) spanning land, maritime-air, fires, medical, logistics and JISR. Adding a format means editing JSON, not code. The loader rejects malformed catalogues.
- **Character codec** (`codec_char.py`): parses and renders a slash-delimited text format (`TYPE/NAME=value/.../` with `//` terminator). It never raises on bad input; it returns a partial message and typed issues.
- **XML codec** (`codec_xml.py`): generates an XSD from the catalogue, emits and validates XML, reads it back. DTD and entity declarations are rejected.
- **Binary codec** (`codec_bin.py`): catalogue-aware compact frame with a 16-bit check. The decoder rebuilds the message, re-validates every field and verifies the check.
- **Delta frames** (`delta.py`): send only changed or removed fields against a baseline both ends hold. The receiver rebuilds the full message, rejects a wrong baseline, and verifies the check on the result.
- **Link emulator** (`link.py`): virtual-time, seeded. Burst loss, periodic outages, shared by forward and return channels. Fragmentation, selective acknowledgement, retransmission with exponential back-off, per-message deadline. One-way mode repeats each fragment and cannot confirm delivery.
- **Workload** (`workload.py`): tracks that send a baseline report then periodic changes (time moves on, one or two fields change).
- **Fusion engine** (`fusion.py`, `geo.py`): resolves reports from several sources to entities using per-family rules in the catalogue (exact identity, request id, or position and time with a motion-aware gate; call-for-fire and MEDEVAC grids are compared in metres, contacts by great-circle distance). Exact resends are ignored; near matches corroborate; attributes such as contact class are decided by a recency-weighted vote in which `UNKNOWN` counts for little, and a close vote is flagged as a conflict instead of being hidden. Where two reports are probably the same thing but the evidence is not strong enough (for example two MEDEVAC requests with different ids at the same pickup), they are kept apart and flagged as a possible duplicate. Every fused field names the report it came from and lists the values it did not choose.
- **Scenario and scoring** (`fusion_sim.py`): a seeded multi-source scenario with ground truth (moving contacts with twins 800 m apart, static targets with twins 300 m apart, MEDEVAC events relayed under other ids, logistics reports arriving out of order, resends, clutter).
- **Free-text extractor** (`extract.py`): drafts a structured message from English or French text. Deterministic and catalogue-driven (keywords, cues, synonyms, units live in the catalogue). Every field has a confidence score and an evidence span; ambiguous values, missing required fields and low-confidence fields are listed as review reasons. It produces a draft for an operator to confirm, never a released message. No machine learning.
- **Generator** (`generator.py`): seeded synthetic messages (English and French text, escaped characters) and seven kinds of text corruption.

## Results: codecs (`demo.py`, seed 42, 400 messages)

| Check | Result |
|---|---|
| Char to message, XML (XSD-valid), char to XML to char, binary round trips | 400 / 400 each |
| Mean bytes: char / XML / XML+zlib / binary | 152.8 / 262.8 / 209.7 / 61.4 |
| Single-bit flips in binary frames | 3,967 of 4,000 detected; 33 harmless (padding bits, same message); 0 changed silently |
| Text corrupted by character substitution | 121 of 400 changed the message with no issue raised |
| Text with one field dropped | 133 of 400 changed the message with no issue raised |

## Results: emulated links (`ddil_demo.py`, 108 messages, mean of 20 link seeds)

Mean wire bytes per delivered message (payload plus fragment headers, retransmissions and acknowledgements):

| Profile | XML | Char | Binary | Delta |
|---|---|---|---|---|
| HF 1.2 kbps, burst loss, 60 s outage / 420 s | 365 | 217 | 91 | 40 |
| HF 2.4 kbps | 365 | 218 | 91 | 41 |
| UHF SATCOM 9.6 kbps | 343 | 202 | 83 | 37 |
| Degraded IP 64 kbps, 10% loss | 392 | 235 | 100 | 44 |
| VLF/LF 200 bps one-way, 3 repeats | 900 | 527 | 212 | 212 |

- In every run, every message the receiver accepted equalled the original (verified = delivered). Messages not delivered by the deadline are counted, not hidden: at HF 1.2 kbps the XML stream lost up to 21 of 108 in the worst seed because the backlog exceeded the 900 s deadline; binary and delta delivered all 108.
- Mean latency on HF is hundreds of seconds because messages queue behind outages and retransmissions; smaller frames shorten the queue (HF 1.2 kbps mean: XML 330 s, binary 142 s, delta 113 s). On faster links latency differences are small.
- One-way broadcast cannot confirm delivery, so delta is not used there (no acknowledged baseline).

## Results: free-text extraction (`extract_demo.py`)

Both sets are hand-written by the developer, English and French, one per line with the expected field values (`data/`).

| Set | Messages | Type correct | Exact messages | Field precision | Field recall |
|---|---|---|---|---|---|
| Development (`handwritten.json`), first run before any fix | 24 | 23 | 19 | 0.982 | 0.931 |
| Development, after fixes | 24 | 24 | 22 | 0.994 | 0.989 |
| Held-out (`holdout.json`), written after the fixes, run once | 12 | 12 | 10 | 0.989 | 0.978 |

What the first run found, and what changed afterwards: a sign bug (a minus sign after a cue such as `lon -62.1` was swallowed, flipping the longitude), a message-type failure when keywords were weak (fixed by falling back to which type's required fields the text fills best), and a missing singular unit (`1 contact`). The singular-unit fix was made after seeing the failure, so the development set is no longer a clean test; the held-out set was written afterwards and not tuned on.

Remaining errors: `fuel` is not a known item (the catalogue lists example items only), an ambiguous precedence ("0 urgent. Routine.") was resolved wrongly but flagged for review, and `Demande EVASAN EVAC9` returned `EVASAN` as the request id at 0.90 confidence. The last one is a confidently wrong value, so confidence is not yet well calibrated at the top.

By language on the development set: English 17 of 19 exact, French 5 of 5. The held-out French set had 3 of 4 exact.

## Results: fusion (`fusion_demo.py`)

Simulated scenario, about 740 reports per run from three sources: 20 moving contacts (plus 4 twins 800 m away and 6 clutter reports), 15 call-for-fire targets (plus 3 twins 300 m away), 10 MEDEVAC events (30% also relayed under a different id), 8 logistics streams with late arrivals, and 8% exact resends. Error rates and noise (position 120 m, class label right 78%, unknown 17%, wrong 5%) are my assumptions. The gate parameters were adjusted after the first run on seeds 42-61; seeds 1000-1019 were run afterwards with no further change.

| | Seeds 42-61 (used while tuning) | Seeds 1000-1019 (fresh) |
|---|---|---|
| Contacts, pair precision / recall | 0.919 / 0.867 | 0.922 / 0.872 |
| Call-for-fservice-monitoring interfaces, UI, a deployable service, security labelling.
- Fusion was tested only on a simulator I wrote, with noise and error rates I chose. It is not a track filter: there is no Kalman filtering, no sensor model and no handling of zone boundaries (grids in different zones are not compared). Treat the numbers as a check that the logic behaves sensibly, not as expected field performan/ 0.969 |
| MEDEVAC and logistics, pair precision / recall | 1.000 / 1.000 | 1.000 / 1.000 |
| Fused contact class: correct / unknown / wrong | 92.9% / 1.9% / 5.1% | 92.8% / 2.1% / 5.1% |
| Fused target type: correct / wrong | 96.4% / 3.6% | 94.8% / 5.2% |
| Fused MEDEVAC security: correct / wrong | 87.0% / 13.0% | 90.0% / 10.0% |
| Latest logistics on-hand correct | 100% | 100% |
| Exact resends ignored | 1,074 of 1,074 | 1,048 of 1,048 |
| MEDEVAC relays under another id flagged as possible duplicates (not merged) | 61 of 61 | 61 of 61 |
| Reports reduced to entities | 14,720 to 1,529 (10%) | 14,755 to 1,512 (10%) |

- Single contact reports are right 78% of the time and `UNKNOWN` 17%. Fusion raises right to 93% and cuts `UNKNOWN` to 2%, but the wrong share does not fall (5% in both): false merges of nearby contacts and wrong votes offset the gain.
- Conflict flags catch only some wrong values: 5 of 36 wrong contact classes, 14 of 20 wrong target types, 4 of 20 wrong MEDEVAC security values (fresh seeds). A single wrong report with no second opinion cannot be detected.
- The contact gate is wide because a DTG has one-minute resolution (a fast contact moves kilometres in that time). Two distinct contacts within about 1-3 km at the same minute will merge, and a track whose reports arrive far apart may split. That is the trade between precision 0.92 and recall 0.87.
- Every report ends up in exactly one entity and every fused value traces to the report that supplied it, in all runs.

## Limits (state these in the proposal)

- Synthetic formats. This is not USMTF, APP-11 or ADatP-3 and makes no conformance claim. Official catalogues are controlled and not furnished.
- Link profiles are assumed, representative values, not measurements of real radios. No operational data and no radios.
- The sender handles one message at a time with back-off, so head-of-line blocking inflates latency. Read the numbers as comparisons between encodings, not as predictions.
- Compression is a catalogue-aware binary encoding compared with zlib, not W3C EXI.
- The 16-bit check detects accidental errors, not tampering. Message authentication is future work.
- Not done: fusion, service-monitoring interfaces, UI, a deployable service.
- Extraction was tested only on messages the developer wrote, knowing the cue lists, so real operator text (typos, shorthand, voice transcripts) will score lower. 36 messages is a small sample; treat the percentages as indicative.

## Licence

Apache License 2.0 (see `LICENSE` and `NOTICE`). Copyright 2026 Alireza Niazi.
