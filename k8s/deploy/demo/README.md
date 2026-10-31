# OpenRL demo app

A small web app for showing OpenRL on a GPU cluster: start a sweep of RL
client Jobs, watch the fleet place them, drill into one GPU or one job over
time, and see the sparse weight deltas a full fine-tuning run publishes.
It runs in the cluster on the API server image with its code mounted from a
ConfigMap, so there is no image to build.

Everything it shows comes from two places:

- the API server's inspection API (`/api/v1/inspect/...`: inventory, runs,
  activity, steps, query), proxied under `/api/inspect/`;
- Kubernetes itself: only the demo's own Jobs and their pods;
- each Job's own log directory on the shared volume, mounted read-only, for
  the client metrics the Metrics tab charts. These are the client's output,
  which the control plane neither holds nor serves.

Nothing is parsed out of logs. A Job's client sets
`openrl.client_log_id=<job name>` in its session tags; the inventory lists
each run's sessions and a session's summary carries its tags, which is how
a Job is matched to its run.

## Tabs

- **Sweep.** A spec editor (tuning mode, model, count, steps, seed) with
  presets. Start submits one client Job per row and count, each running the
  repo's `fft-gsm8k-rl` or `lora-gsm8k-rl` scenario with
  `TINKER_TAGS=openrl.recipe=rl`. Stop releases the run's workers through
  `delete_model` and deletes the Job. Jobs are listed newest first; sweeps get
  a letter (A is the newest) so job numbers stay unambiguous across sweeps.
- **Fleet.** Counts, the job table with step progress, one tile per GPU with
  a chip per worker holding it, the waiting lane, GPU time held since the
  first submit, and DCGM duty when the API server has a metrics source.
  Click a tile or a job row to open it on the Timeline tab.
- **Timeline.** One occupancy trace per GPU: 1 while a worker is on it,
  coloured by the holder, 0 when idle; the thin band at the base is a worker
  that holds the GPU between time-slicer turns. Live view scrolls; drag on
  the trace to select a window, and the shares are computed over exactly that
  span. For a job, the overlay toggle adds the other workers on its GPUs.
- **Weights.** For a full fine-tuning run, the share of the model's weights
  that changed in each sparse delta the trainer published after an optimizer
  step, from the save op's metrics in `/runs/{id}/steps`
  (`delta/density_pct`).

- **Metrics.** What each job's client measured per step: accuracy, reward
  and output length, one line per job in its colour with the sweep's mean on
  top, or one job alone. Read from the `metrics.jsonl` the client writes to
  the log directory the app assigned it (`RUNS_ROOT/<job>`).

## Deploy

```bash
kubectl apply -k k8s/deploy/demo
kubectl port-forward svc/open-rl-demo 8090:8000
open http://127.0.0.1:8090/
```

The bundle expects an OpenRL installation in the same namespace
(`k8s/deploy/fft` or `k8s/deploy/lora`). It runs as that installation's
service account, `open-rl-sa`, and binds a Role to it for the Jobs it
creates and the pod logs it reads. Point `CLIENT_IMAGE` at the client
image that matches the server, and set `CLUSTER_NAME` if you want it on the
Fleet tab. The Timeline's turn-level trace and the GPU tiles need the API
server to report worker turns and node GPUs in the inspection API; without
them the trace falls back to placement holds and the tiles stay empty. The
Weights tab needs the save op's delta metrics in the run's steps. The DCGM
duty figure needs `OPEN_RL_PROMETHEUS_URL` on the API
server.

## Files

- `app.py`: FastAPI backend (Jobs, the proxy, the presets).
- `index.html`: the page; canvas rendering, no build step.
- `demo-app.yaml`: ServiceAccount, Role, Deployment, Service.
