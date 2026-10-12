# Connecting

## URLs

| From | URL |
|---|---|
| A laptop | `kubectl -n openrl-system port-forward svc/open-rl-api-server-service 9003:8000` in a loop, then `http://127.0.0.1:9003` |
| A pod in the cluster | `http://open-rl-api-server-service.openrl-system.svc.cluster.local:8000` |

A port-forward loop that survives server restarts:

```bash
while true; do kubectl -n openrl-system port-forward svc/open-rl-api-server-service 9003:8000; sleep 1; done
```

The SDK reads `TINKER_BASE_URL`, or pass `base_url=` to `tinker.ServiceClient`. The server
does not check credentials; set `TINKER_API_KEY` to any string because the SDK refuses to
start without one.

## Which server you are talking to

```bash
curl -s http://127.0.0.1:9003/api/v1/get_server_capabilities | python3 -m json.tool
```

The `openrl` object holds `version` (the release tag, or `sha-<commit>` for a build from
main), `worker_launcher` (`scheduler` on Kubernetes, `process` or `single_process` for local
servers), `sampler_backend`, `fft_enabled`, `models`, `settings` (the same table as
`settings.md`), `limits`, `endpoints` and, on releases, `skill_url` for the copy of this
skill that matches the server. A response without `openrl` is a server older than v0.0.7.

## OpenRL-only endpoints

Not part of the Tinker API; plain HTTP GET/POST with JSON.

| Endpoint | Use |
|---|---|
| `GET /api/v1/openrl/status` | every worker this server runs, with Workload phase, reason, node, pod state and age; live sessions; the fleet: accelerator type and memory per node and how many run an OpenRL worker |
| `GET /api/v1/openrl/models` | every model created, newest first, with owner and whether its session is live |
| `GET /api/v1/openrl/models/{model_id}` | one model: settings, sessions, workers, and `summary`, one line that says what to know first |
| `GET /api/v1/openrl/sessions/{session_id}` | live or not, TTL, settings the session was opened with, owners attached |
| `POST /api/v1/delete_model` | `{"model_id": "..."}`; releases the model's workers now (an FFT or exclusive job) or frees its adapter (a shared LoRA runtime) |
| `GET /openapi.json`, `/docs` | the full API |

The SDK's training client gives you the model id as `trainer._guaranteed_model_id()`;
the session id is not exposed, read it from the model view's `sessions`.

## SDK versions

The repository's examples pin `tinker==0.29.0` and `tinker-cookbook[math-rl]==0.5.7`.
Servers from v0.0.4 on speak the protobuf wire SDKs >= 0.25 use; older SDKs also work.
The SDK's response models ignore unknown fields, so the `openrl` additions are safe for
every version.

## Supported Tinker surface

From `docs/tinker-client-compatibility.md`, generated against the server:

- `ServiceClient`: `create_lora_training_client`, `create_sampling_client`,
  `create_training_client_from_state` (with and without optimizer), `get_server_capabilities`.
- `TrainingClient`: `forward`, `forward_backward`, `optim_step`, `save_state`, `load_state`
  (with and without optimizer), `save_weights_for_sampler`,
  `save_weights_and_get_sampling_client`, `create_sampling_client`, `get_info`, `get_tokenizer`.
- `SamplingClient`: `sample`, `compute_logprobs`, `get_base_model`, `get_tokenizer`.

Not supported: the `RestClient` checkpoint listing, deletion, archive and publish calls, and
the session, billing and audit APIs. A cookbook recipe that lists checkpoints to resume will
not find any; pass `save_every=0` unless you have verified resume on your server.

Full fine-tuning is a setting, not a different client: `openrl.fine_tuning_type=full` on the
session or the model. The old `x-open-rl-fine-tuning-type` header is refused with a 400
that names the setting.
