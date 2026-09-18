# AdamLM 2.0 — Implementation Progress Record

Purpose: resume without repeating completed steps. No training, no provider
calls, no promotions, no deletions performed by any step below.

- [x] Phase 0: pointer backup `results/assistant_default.20260918-020742.bak.json`
      verified byte-identical to live pointer. No git repo (stays that way).
- [x] Phase 1: skill-profile key fix, unified auto/manual gate (probes + suite
      0.02 tolerance + reported val-loss), preflight budget block + Budget UI
      row, plan Phase-0 check fixed. Tests green (59 passed, 2026-09-18).
- [x] Phase 5: providers.py seam (OpenRouter moved verbatim; groq/unorouter/
      apinex/xkiro/ollama reserved, rejected). research.py re-exports aliases.
      Stored `max_api_requests` 1.9e21 -> 25 (+`provider: openrouter`). Key untouched.
- [x] Web endpoints: `POST /api/research/preview` (pure), session create fields,
      `POST /api/versions/archive`, `/api/versions` via read_registry (pure fallback).
- [x] New tests: research controls, preview purity, archive, provider guards
      (test_one_adamlm.py 31 passed).
- [ ] Frontend: research tab, versions tab, renderResearch fix, CSS.
- [x] Frontend: research dataset/stage/repetition controls + preview panel +
      resume; versions groups/filter/archive + promote confirm; renderResearch
      `box` fix; chat failure next-step hint + capability footnote; CSS for new
      controls + 1700px breakpoint. node --check + JS OK.
- [x] Training-tab repeat confirmation gate for capped (multi-pass) plans.
- [x] Integration: full suite 135 passed; live scratch-port checks for
      /api/versions (archived), /api/research/preview (pure), archive cycle,
      /api/overview. Prod results/ restored to 18 files; pointer hash unchanged.
- [x] Cleanup: removed my smoke-test backend tree (31280/35868 left running —
      pre-existing user backend, untouched).
- [ ] Final: user report (below in chat).

Backups / safety notes:
- Approved pointer + checkpoints untouched (see Phase 0 hashes in session log).
- `config/.research-api-key` never read into responses; key code paths unchanged.
