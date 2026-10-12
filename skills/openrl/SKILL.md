---
name: openrl
description: Use a deployed OpenRL server (a self-hosted, Kubernetes-native Tinker API for LoRA and full fine-tuning, SFT and RL) from the client side with the tinker SDK or tinker-cookbook. Covers connecting, the openrl.* settings, what the cluster is doing while a call blocks, and what each error means. Use when a task mentions OpenRL, open-rl, the OpenRL API server, "our Tinker endpoint", or debugging training runs against one. Not for operating the cluster or for the hosted Tinker service.
---

# OpenRL from the client side

OpenRL release: dev

OpenRL (github.com/gke-labs/open-rl) serves the Tinker API on a Kubernetes
cluster your platform team runs. Any `tinker` SDK client or `tinker-cookbook`
recipe works against it; your training loop runs on your machine and the
GPUs do the work on the cluster. Facts here were verified on v0.0.6 plus the
status endpoints added by design 016, on 2026-10-12. The line above is
replaced with the release tag when the skill ships with a release.

## Quick card

| | |
|---|---|
| Endpoint from a laptop | `kubectl -n openrl-system port-forward svc/open-rl-api-server-service 9003:8000`, then `http://127.0.0.1:9003`. Run it in a loop; it drops on server restarts (see runbook). |
| Endpoint in the cluster | `http://open-rl-api-server-service.openrl-system.svc.cluster.local:8000` |
| Auth | None. The SDK insists on a key: `TINKER_API_KEY=tml-dummy-key`. |
| Is it up, which release | `GET /api/v1/healthz`; `GET /api/v1/get_server_capabilities` -> `openrl.version`, `openrl.worker_launcher`, `openrl.fft_enabled`, `openrl.settings` |
| What is the cluster doing | `GET /api/v1/openrl/status` (every worker, its phase and node, free accelerators); `GET /api/v1/openrl/models/<model_id>` (one model's workers, sessions, a one-line summary) |
| One-command check | `uv --project examples run openrl-doctor` from a clone, or `python -m common.openrl_doctor` in the client image |
| Base models | Any Hugging Face model id. `supported_models` lists only the configured default. First use of a model downloads it on the cluster. |
| LoRA or full fine-tuning | `openrl.fine_tuning_type=lora` (default) or `full`; full needs `openrl.fft_enabled: true` in capabilities |
| Settings | `references/settings.md`; pass as `user_metadata` or `TINKER_TAGS=openrl.key=value` |
| Keep the run alive | Keep the client process and the port-forward up. Session TTL is 120 s; workers are released about 2 minutes after the last heartbeat. |
| Release workers now | `POST /api/v1/delete_model {"model_id": ...}` (not in the SDK) |

## Start here

1. **Check the deployment.** `openrl-doctor`, or curl `healthz`, `get_server_capabilities`
   and `/api/v1/openrl/status`. Read `openrl.version`; a server without an `openrl` section
   predates v0.0.7 and has no status views.
2. **Pick the settings.** `references/settings.md` is the complete list with defaults.
   Most runs need none. Full fine-tuning, exclusive GPUs, more sampler replicas and the
   Automodel trainer are settings, not code.
3. **Write or run the loop.** `examples/tiny/tiny_sft.py` and `tiny_rl.py` in the repository
   are complete; cookbook recipes run unchanged with `base_url=` pointing at the server.
4. **Know what waiting looks like.** `references/lifecycle.md`: which worker starts when,
   how long each phase normally takes, and when workers go away.
5. **When something is wrong**, start from the symptom in `references/runbook.md`. Every
   entry says what to check from the client, the fix, and what to tell the platform owner.

## Rules that save runs

- Read `summary` from `/api/v1/openrl/models/<id>` before deciding a run is hung. The
  examples print it every minute while a call blocks.
- A port-forward that is down for more than two minutes ends the run: the server releases
  the workers and the adapter state in them. Save state (`save_state`) if a run must survive
  that, and restart the forward before relaunching.
- Call `delete_model` when a script exits normally, so the next run does not wait for the reaper.
- Do not delete Workloads or pods under a live run; that strands the client (issue #246).
- Settings are strings on the wire: `"true"`, `"2"`.

## Files

| Path | Contents |
|---|---|
| `references/settings.md` | every `openrl.*` setting, generated from the server code and tested |
| `references/connect.md` | URLs, SDK versions, the supported Tinker surface, OpenRL-only endpoints |
| `references/lifecycle.md` | workers, Workload phases, normal durations, release and cleanup |
| `references/runbook.md` | symptom -> check -> fix -> what to tell the platform owner |
