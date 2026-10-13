# Workers and their lifecycle

Numbers are from a GKE cluster with L4 and H100 nodes on 2026-10-12, with the worker
image already on the nodes and the model already in the shared cache. A cold node adds
an image pull (the worker image is several GiB) and a model download.

## Who runs what

| Your call | What starts | Shared with |
|---|---|---|
| `create_lora_training_client` (LoRA, default) | one trainer for the base model, Workload `lora-<base>-0-trainer` | every other LoRA job on the same base model; each job is one adapter in it |
| first `sample` or `save_weights_and_get_sampling_client` | one sampler, `lora-<base>-0-sampler` (plus `-1`, `-2`... with `openrl.sampler_replicas`) | same |
| `create_lora_training_client` with `openrl.fine_tuning_type=full`, or `openrl.exclusive=true`, or `openrl.trainer_gpus>1` | a trainer of its own, `fft-<model_id>-trainer`, and later `fft-<model_id>-sampler` | nothing; full fine-tuning workers take turns on a GPU with other jobs' workers unless exclusive |

The workers that serve one job appear together in the job's model view,
`GET /api/v1/openrl/models/{model_id}`.

## Worker states

Each worker in the model view has a `state`, the same five values on every server, and a
`detail` in the scheduler's own words (Workload phase and reason, node). Branch on `state`;
show `detail` to a human. `kubectl -n openrl-system get workloads` shows the same facts
`detail` quotes; the pods behind them are the scheduler's and the view does not read them.

| State | Meaning | What `detail` says | Normal duration |
|---|---|---|---|
| `waiting_for_placement` | no accelerator assigned yet | `Pending (NoCapacity: ...)`: nothing in the cluster fits; `Pending (WaitingForCapacity)`: something fits but is busy | seconds; longer only when the cluster is full |
| `starting` | placed on a node; pulling the image, mounting volumes, or loading the model | `Placing (PodCreated) on <node>` is the image pull and volume mount; `Running ..., loading the model` is the trainer loading the base model or vLLM starting | under a minute on a warm node; 5 to 10 minutes on a cold node. Trainer: `ready` about 2 minutes after `create_model`; sampler: 3 to 4 minutes more while vLLM starts |
| `ready` | serving requests: the worker has its model loaded and is draining its queue | node name | |
| `failed` | the process exited with an error, or the request cannot be satisfied | the scheduler's reason, for example `Failed (PodFailed)`. A container that cannot pull its image or crash-loops is not reported as failed yet; it stays `starting` or `ready` past the normal duration, see the runbook | |
| `stopped` | exited normally, or being released | `being released; ...` while the reaper or `delete_model` tears it down | |

What the examples print while they wait is the same information:

```
[tiny-sft] waiting 2m00s on the first sample (the cluster is starting a sampler); trainer ready: Running (PodCreated) on gke-...-lvw8, 201s; sampler starting: Running on gke-...-2208, loading the model, 120s
```

So, from a laptop against a warm cluster: about 2 minutes to the first loss line and about
4 more minutes to the first sample. A second run on the same base model within two minutes
of the first finds both workers up and starts in seconds.

## Sessions and release

- The SDK heartbeats its session every 10 seconds. The server keeps a session alive for
  120 seconds after the last heartbeat (`limits.session_ttl_seconds` in capabilities).
- A reaper runs every 30 seconds and releases every worker whose sessions are all gone.
  Observed: workers gone 2 minutes 7 seconds after the client exited.
- Release deletes the Workloads and pods. Adapter weights that were only in the trainer's
  memory are gone with it; checkpoints written by `save_state` or `save_weights_for_sampler`
  stay on the shared volume and can be restored with `create_training_client_from_state`.
- A heartbeat gap longer than the TTL has the same effect as exiting, including a
  port-forward that is down for two minutes. On servers up to v0.0.6 the client is not told;
  it keeps polling `try_again`. The model view then says
  `no live session and no workers; the model's workers were released`.
- `POST /api/v1/delete_model` releases right away, or frees the adapter in a shared runtime.

## Sharing and speed

- LoRA jobs on one base model take turns in one trainer, so a step takes longer when several
  run at once; their sampling overlaps with training.
- Full fine-tuning jobs each get a trainer; on the FFT release they time-slice a GPU, parking
  state in host memory between turns, so concurrent FFT jobs slow each other down.
- A sampler that is still loading a model is `starting` with a `detail` that ends in
  `loading the model`; requests queue until it is `ready`.
