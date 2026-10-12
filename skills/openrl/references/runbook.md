# Runbook: symptom -> check from the client -> fix -> what to tell the platform owner

Check first, always: `GET /api/v1/openrl/models/<model_id>` and read `summary`. Compare
worker ages with the durations in `lifecycle.md` before calling anything hung. Without a
model id (the client is still inside `create_model`) the only view is
`kubectl -n openrl-system get workloads,pods`, which needs cluster access; without that,
wait out the normal duration, then ask the platform owner.

| Symptom | Check | Fix | Tell the platform owner |
|---|---|---|---|
| No output for minutes after `create_model` or the first `sample` | the worker's `state`, `detail` and age in the model view; the examples print this every minute once they have a model id | wait while `waiting_for_placement` or `starting` and younger than the normal duration; a cold node needs 5 to 10 minutes more | nothing |
| Worker `waiting_for_placement`, `detail` `Pending (NoCapacity: ...)` | the reason names what is missing: a node accepting the role, or a device with enough memory | a smaller model, `openrl.trainer_gpus` if the model needs more than one device (Automodel only), or wait for the owner | the reason text; which nodes should carry `openrl.io/enabled=true` and the role labels |
| Worker `waiting_for_placement`, `detail` `Pending (WaitingForCapacity)` | nothing more from the endpoint: the fleet is not visible to clients yet | wait; idle workers of other jobs are released about two minutes after their clients exit | how long you have waited; `kubectl -n openrl-system get workloads` shows them who holds the GPUs |
| `starting` for over 10 minutes, `detail` says `pod ContainerCreating` | `detail` and `age_seconds` | wait; it is pulling the worker image or mounting the shared volume | the entry's `node` and `workload` if it exceeds 15 minutes |
| Worker `failed` | `detail`: the reason, the pod's waiting reason and restart count; a sampler failing on a small GPU usually ran out of memory | smaller model, or a larger device via `openrl.exclusive` on a bigger node pool | the `workload` name; the owner reads the pod log |
| Client logs `Retrying request ... because of try_again` for many minutes; model view says `no live session and no workers; the model's workers were released` | `session_live: false` in the model view: the session expired once | the run is lost on servers up to v0.0.6. Stop the client, make sure the port-forward is up, start a new run; restore from the last `save_state` path if there is one | time of the gap, so they can see whether the server restarted |
| `Session heartbeat failed for N seconds` in the client log | `curl .../healthz` through the same URL the client uses | the connection, not the server: restart the port-forward loop. Under 120 s the run continues; over 120 s see the row above | nothing, unless `healthz` itself fails |
| HTTP 400 `The x-open-rl-fine-tuning-type header was removed; set openrl.fine_tuning_type ...` | the client sets a removed header or `full_config` | pass the setting named in the message as `user_metadata` or `TINKER_TAGS` | nothing |
| HTTP 400 `openrl.<name>: unknown setting` or `must be ...` | spelling and value against `settings.md` | fix the value; values are strings | nothing |
| HTTP 400 `openrl.exclusive needs a server that launches workers per model` (or `sampler_replicas`, `trainer_backend`, `trainer_gpus`) | `openrl.worker_launcher` in capabilities is not `scheduler` | run against the Kubernetes deployment; these settings need it | nothing |
| HTTP 400 on `create_model` with `openrl.fine_tuning_type=full` saying FFT is disabled | `openrl.fft_enabled` in capabilities | use LoRA, or ask for the FFT release | "please deploy `openrl-fft.yaml`" |
| Samples are token soup or a different model's style | `get_info` for your model returns your base model as `tokenizer_id` | report; this was a server bug fixed in v0.0.5 | OpenRL version and the model id |
| Steps are two to three times slower than yesterday | `shares_workers: true` in the model view means other LoRA jobs on the same base model share your trainer; several FFT jobs time-slice a GPU | expected sharing; stagger runs if wall-clock matters | nothing |
| Everything slow or `healthz` failing | `healthz` latency | stop clients and wait | the `healthz` output and time |
| `No module named cluster_wait` when running a tiny example | you ran it from a copy, not from `examples/tiny/` | run from the repository clone, or copy `cluster_wait.py` next to the script | nothing |

## Where the server-side evidence is, for the platform owner

- API server: `kubectl -n openrl-system logs deploy/open-rl-api-server`. Worker release
  lines read `No live session uses <owner>; tearing its workers down`.
- Workers: pods named `orw-<workload>`, deleted on release, so a post-mortem needs the
  cluster's log store (Cloud Logging on GKE).
- Workloads: `kubectl -n openrl-system get workloads` shows the phases and reasons the
  model view's `detail` quotes, for every job at once.

## Cleanup hygiene

- Normal exit: call `POST /api/v1/delete_model` so the workers go now instead of in two minutes.
- A killed client: nothing to do; the reaper releases its workers.
- Never `kubectl delete workload` under a live run; the client is not told and polls forever
  (issue #246).
