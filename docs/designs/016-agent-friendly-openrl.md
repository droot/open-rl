# Design Doc 016: Making OpenRL agent friendly

**Status:** Draft
**Author:** Sunil Arora
**Last updated:** 2026-10-12

## 1. Summary

Most OpenRL users will drive it through an AI coding agent: the agent reads
the docs, writes the training loop, runs it, and debugs it. An agent is a
user that cannot ask a colleague, so everything it needs has to be
discoverable from the endpoint it is connected to and from the files in the
repository.

Today it is not. During the v0.0.6 release test (`openrl-release-test/v0.0.6/FINDINGS.md`),
an agent following `docs/getting-started.md` had to read the server source,
`kubectl get workloads`, and `kubectl exec` into Redis to find out why a run
hung without an error. This document proposes:

1. **A self-describing endpoint.** `get_server_capabilities` reports the
   release, the `openrl.*` settings with their types and defaults, the
   worker launcher mode, and limits. The OpenAPI document gets a real title
   and version.
2. **Status without kubectl.** New read-only endpoints under
   `/api/v1/openrl/` return what the API server already knows: a model's
   workers and their Workload phase and reason, live sessions, queue depth,
   and a fleet summary.
3. **Every failure reaches the client, with the fix in the message.** When
   workers are released, pending futures fail instead of polling forever.
   While a worker is starting, `try_again` says what it is waiting on.
4. **A reference written for agents.** A skill (`skills/openrl/`) with the
   settings table, Workload phases and normal durations, and a
   symptom-first runbook. The settings table is checked against the code
   by a test.
5. **A doctor script and parseable examples.** `openrl-doctor` verifies a
   deployment in under a minute; the tiny examples flush output and exit
   non-zero on failure.

All API changes are additive. No Tinker wire path changes shape.

## 2. Evidence

What the agent needed during the v0.0.6 test, and where it found it:

| Question | Where the answer was | Where it should be |
|---|---|---|
| Why has the run shown nothing for 8 minutes? | `kubectl get workloads`; then `session_registry.py`, `api_server.py`; then `redis-cli` inside the Redis pod | The client's `try_again` response, or an error |
| What are the `openrl.*` settings and their values? | Release notes and `class Settings` in `api_server.py` | `get_server_capabilities` and the skill |
| Does this server support gemma? | Unclear: `get_server_capabilities` listed only `Qwen/Qwen2.5-0.5B`, the guide trains gemma | `get_server_capabilities` |
| Is the trainer pulling an image or stuck? | `kubectl -n openrl-system get pods` | A status endpoint |
| Which release is deployed? | `kubectl get deploy -o jsonpath=...` | `get_server_capabilities`, `/openapi.json` |
| Is the client hung or buffering? | Guess; `ps` and `lsof` | Flushed output |

The hang itself (FINDINGS F1): a port-forward dropped, heartbeats failed
for longer than the 120 s session TTL, the reaper released the workers,
and the heartbeat then recreated the session key without its owner
binding. The `create_model` future was never resolved, so the client
polled `try_again` indefinitely. The API server log filters heartbeats and
`retrieve_future`, so nothing recorded the revival. Issues #246 and #247
describe the same class of failure.

## 3. Goals and non-goals

Goals:

- An agent with only the HTTP endpoint can learn the deployment's version,
  settings, limits and current state, and gets an actionable error for
  every failure the API server can detect.
- An agent with the repository checked out finds a reference it can load
  as a skill, kept honest by tests.
- A human reading the same endpoints and docs is served equally well.
- No change to Tinker SDK compatibility. Existing clients, including
  `tinker-cookbook`, keep working unchanged.

Non-goals for this round:

- An MCP server. REST plus OpenAPI plus a skill is enough for every agent
  that can call HTTP. Revisit when there are operations that do not map
  to the API, such as streaming worker logs.
- Authentication or per-user authorization. The new endpoints expose
  cluster details (node names, pod names) to anyone who can reach the API
  server, which is already true of the API as a whole.
- Changing the session TTL or the reaper policy (#247). This document makes
  the consequences visible; the policy is a separate decision.
- A log streaming endpoint.

## 4. Principles

1. **The endpoint is the documentation.** If an agent would have to read
   source to learn a fact about the running server, the server should
   report it.
2. **Silence is a bug.** A request that cannot be served fails with a
   message that names the cause and the next step. Waiting is fine;
   waiting without saying why is not.
3. **Status needs no cluster access.** Most agents run on a laptop behind
   a port-forward. Everything `kubectl get workloads,pods` shows about a
   model, the API server can show too.
4. **Docs for agents are references, not narratives.** Tables, exact
   commands, normal durations, and symptom-first runbooks.
   `getting-started.md` stays a narrative for humans.
5. **Facts in docs are tested.** A settings table that drifts from
   `class Settings` is worse than none.

## 5. Design

### 5.1 Self-describing endpoint

`GET /api/v1/get_server_capabilities` keeps its current fields and gains an
`openrl` object. The SDK's response models ignore unknown fields
(`extra="ignore"` in `tinker/_models.py`), so this is safe for every SDK
version OpenRL supports.

```json
{
  "supported_models": [{"model_name": "Qwen/Qwen2.5-0.5B"}],
  "default_model": "Qwen/Qwen2.5-0.5B",
  "single_process": false,
  "openrl": {
    "version": "v0.0.6",
    "build": "1f08142",
    "worker_launcher": "scheduler",
    "sampler_backend": "vllm",
    "fft_enabled": true,
    "models": {
      "any_huggingface_model": true,
      "examples": ["google/gemma-4-e2b", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B"]
    },
    "settings": [
      {"name": "openrl.fine_tuning_type", "type": "string", "default": "lora",
       "allowed": ["lora", "full"], "description": "LoRA unless set. A restored model takes the checkpoint's type."},
      {"name": "openrl.exclusive", "type": "bool", "default": false,
       "description": "A workload no other workload shares. ..."}
    ],
    "limits": {"vllm_max_model_len": 8192, "train_token_budget": 65536, "session_ttl_seconds": 120},
    "endpoints": {"status": "/api/v1/openrl/status", "models": "/api/v1/openrl/models", "openapi": "/openapi.json"}
  }
}
```

- `settings` is generated from `class Settings`: field name, annotation,
  default, `Literal` or enum choices, and the comment above the field,
  which becomes the description. One source of truth for the endpoint,
  the skill, and the 400 messages.
- `worker_launcher` is one of `single_process`, `process`, `scheduler`,
  derived from the same environment the server already reads. Several
  settings are refused outside `scheduler` mode; the client can see why.
- `models`: in `scheduler` and `process` modes any Hugging Face model
  works and `BASE_MODEL` is only a default. `supported_models` stays as
  the SDK expects it; `models.any_huggingface_model` says the truth, and
  `examples` lists models the examples and tests use.
- `version` comes from a build argument `OPEN_RL_VERSION` baked into the
  image and exposed as an environment variable. `/app/VERSION` inside the
  v0.0.6 image reads `0.6.18`, the internal counter, so it cannot serve as
  the release version. The release workflow passes `GITHUB_REF_NAME`; dev
  builds get the short commit.
- `limits` reports the environment values the server already forwards to
  workers (`VLLM_MAX_MODEL_LEN`, `OPEN_RL_TRAIN_TOKEN_BUDGET`) and the
  session TTL.

The FastAPI app metadata changes from `Open-RL Server MVP 0.1.0` to
`OpenRL API server`, the release version, and a description linking the
repository and the skill. `/openapi.json` and `/docs` already exist; this
makes them worth reading. Endpoint docstrings are already good; a few
request models gain field descriptions.

### 5.2 Status endpoints

New read-only endpoints, all under `/api/v1/openrl/`, all JSON, none used
by the Tinker SDK:

| Endpoint | Returns |
|---|---|
| `GET /api/v1/openrl/status` | `version`, `worker_launcher`, store reachable, live session count, owners with workers, Workload counts by phase, fleet summary |
| `GET /api/v1/openrl/models` | Every model the store knows with `base_model`, `fine_tuning_type`, owner, whether its session is live, worker phases |
| `GET /api/v1/openrl/models/{model_id}` | The model's metadata and resolved settings, its sessions, and one entry per worker (below) |
| `GET /api/v1/openrl/sessions/{session_id}` | Live or expired, TTL remaining, owners attached, user_metadata |

A worker entry:

```json
{
  "role": "trainer", "index": 0,
  "workload": "lora-google-gemma-4-e2b-0-trainer",
  "phase": "Placing", "reason": "PodCreated",
  "pod": "orw-lora-google-gemma-4-e2b-0-trainer", "pod_phase": "Pending",
  "pod_waiting": "ContainerCreating",
  "node": "gke-open-rl-dra-l4-trainer-2x-ed601d8b-lvw8",
  "age_seconds": 128,
  "ready": false
}
```

Sources: the Workload's `status.phase`, `status.reason`, `status.podName`,
`status.nodeName`, and `metadata.creationTimestamp`; the pod's phase and
first container waiting reason; `open_rl:sampler_ready:<model>` for
samplers. The API server's service account already has `get`, `list`,
`watch` on Workloads (namespaced), and on pods, nodes and ResourceSlices
(cluster-scoped, `open-rl-fleet-reader`), so no RBAC change is needed.

The fleet summary in `/status` counts devices per accelerator type from
ResourceSlices and how many are claimed, using the same reads the fleet
reader role was created for. It answers "is there a free GPU" without
`kubectl`. It is best-effort: if the reads fail, the field is absent and
`status` says why.

In `process` and `single_process` modes the endpoints return the same
shape with the worker fields the process launcher knows (pid, alive).

### 5.3 Failures reach the client

Three changes in the API server, each covered by a unit test in
`tests/test_session_lifecycle.py`:

**Releasing workers fails their pending futures.** `reap_owner` and
`delete_model` already know the owner's model ids. Before deleting the
Workloads, every unresolved future for those models is set to
`RequestFailedResponse` with a message like:

> workers for model `df85...` were released: no live session for 120 s
> (last heartbeat 2026-10-12T00:14:51Z). Create a new model, or resume
> from `tinker://df85.../weights/step-9` if you saved one. Keep the client
> process running and the port-forward up; see
> `/api/v1/openrl/models/df85...`.

Queued commands for those models are drained at the same time, so a worker
started later for the same owner does not execute stale work.

**A heartbeat after expiry does not revive the session.** An unknown
session id gets a `200` whose body carries `openrl.expired: true`, the key
is not recreated, and one unfiltered log line records it. Requests that
name an expired session get a 400 that says so and points at
`/api/v1/openrl/sessions/{id}`.

**`try_again` says what it is waiting on.** While a future is pending,
the poll response gains an `openrl` object: the stage (`launching
trainer`, `queued`, `running`), the worker's Workload phase and reason,
and how long the request has waited. If the Workload is `Pending` with a
`NoCapacity` reason for longer than a configurable limit (default 10
minutes), the future fails with that reason. `WaitingForCapacity` keeps
waiting, because the scheduler expects a seat to free up.

### 5.4 Logs

The `FilterNoisyEndpoints` filter stays, since heartbeats and polls are
noise. Session expiry, revival attempts, worker release, and every future
the server fails get one structured line each, so a post-mortem from the
API server log alone is possible.

### 5.5 A reference for agents

`skills/openrl/SKILL.md` plus `references/` in this repository, loadable
by Claude Code and compatible agents, and linked from
`docs/getting-started.md` and the README. Contents:

- Connect: the endpoint, the port-forward, what to check first
  (`healthz`, `get_server_capabilities`, `/api/v1/openrl/status`).
- Settings table, generated from `class Settings` by
  `scripts/gen_settings_doc.py`; a unit test fails when the checked-in
  table is stale.
- Workload phases and normal durations: `Pending` to `Running` for a cold
  node (image pull), a warm node, and a reused worker; vLLM start time;
  the 120 s idle release.
- Runbook, symptom first: no output for N minutes, `NoCapacity`,
  `WaitingForCapacity`, a 400 naming a setting, a released-workers error.
  Each entry: what to check from the client, the fix, what to tell the
  platform owner.
- Lifecycle rules: keep the client process alive, keep the port-forward
  up, call `delete_model` at exit, save checkpoints if a run must survive
  a disconnect.

The earlier draft at `~/work/lora-without-regret/skills/openrl` (verified
on v0.0.4) is the starting point; its facts are re-verified against the
current release and its gateway-era names are updated.

**Shipped with every release.** `make release-bundle` adds
`openrl-skill.tar.gz`, the contents of `skills/openrl/` at the tag with
the "verified on" stamp set to the release version, listed in
`checksums.sha256` like the manifests. The repository copy stays
canonical; the asset is a copy, so a clone at the tag and the asset are
the same thing. `get_server_capabilities` reports
`openrl.skill_url` for the running version
(`https://github.com/gke-labs/open-rl/releases/download/<version>/openrl-skill.tar.gz`),
so an agent that starts from an endpoint finds the matching reference in
one call. Claude Code users get a second path, a `.claude-plugin/marketplace.json`
so `/plugin marketplace add gke-labs/open-rl` works and can pin a tag;
the tarball covers other agents, scripted installs, and air-gapped
clusters. Installing is still a manual step for every agent, so the
getting-started guide gets one install command right after
`kubectl apply`. `RELEASING.md` gets a step to re-read the timings and
runbook, since only the settings table is checked by a test.

**Discovery.** Agents fetch the page they are pointed at and follow its
links; none probes `/llms.txt` on its own. So the landing page and the
README each get a plain link to the skill near the top, and a
`site/llms.txt` is added as a convenience for anyone told to read it.

### 5.6 Doctor and examples

`openrl-doctor` (a script in the examples environment, also copied into
the client image) runs in under a minute without a GPU:

1. `healthz` and `get_server_capabilities`; prints version, launcher mode,
   settings count.
2. `/api/v1/openrl/status`; prints free devices per accelerator type and
   anything already running.
3. Optional `--train`: a 3-step `tiny_sft` round trip with cleanup, timing
   each stage, so an agent learns what "normal" takes on this cluster.

One line per check, `OK`/`WARN`/`FAIL` with the reason, non-zero exit on
`FAIL`.

The tiny examples switch stdout to line buffering, exit non-zero on
failure, silence the deprecation and Hugging Face warnings, and when a
call has waited more than 60 s print one line from the model's status
endpoint ("waiting on cluster: trainer Placing (PodCreated), 2m10s").

## 6. Decisions

| Decision | Recommendation | Why | Status |
| --- | --- | --- | --- |
| API change scope | Additive only: new fields nested under `openrl`, new paths under `/api/v1/openrl/` | SDK response models ignore unknown fields; Tinker wire paths stay identical | open |
| Where OpenRL-specific data lives in Tinker responses | One `openrl` object | Easy to find, easy to strip, never collides with SDK fields | open |
| Settings documentation | Generated from `class Settings`, checked by a test | One source of truth; the 400 messages already derive from it | open |
| Version source | Build arg `OPEN_RL_VERSION`, env var at runtime | `/app/VERSION` holds the internal counter, not the release tag | open |
| Expired session semantics | No revival; 200 with `expired: true` on heartbeat, 400 on other calls | Avoids a live-looking session with no workers; keeps the SDK heartbeat loop from erroring | open |
| Agent reference location | `skills/openrl/` in this repository, copied into each release as `openrl-skill.tar.gz` | Versioned with the code the tests check it against; the asset matches the manifests a user deployed | decided |
| Expired session semantics | No revival; 200 with `expired: true` on heartbeat, 400 on other calls | Avoids a live-looking session with no workers | decided |
| Version source | Build arg `OPEN_RL_VERSION`, env var at runtime | `/app/VERSION` holds the internal counter, not the release tag | decided |
| MCP server | Not now | REST plus skill covers current agents | decided |
| Fleet summary from ResourceSlices | Best-effort field in `/status` | Uses existing RBAC; absence is reported, not fatal | open |

## 7. Phases

**Phase 1: see what is happening.** Implemented on this branch and
verified on 2026-10-12 against `open-rl-dra` (api-server image
`agent1-27de66b` over the v0.0.6 bundle): `openrl-doctor` reports the
fleet, `tiny_sft` prints the cluster's state while it waits, and a client
stranded by a port-forward outage shows up as "no live session and no
workers; the model's workers were released" in its model view. API server
only, no cluster or RBAC change. Capabilities `openrl` object and OpenAPI metadata (5.1); the four
status endpoints (5.2); build arg for the version; tiny examples flush and
exit codes; a first `openrl-doctor` using the new endpoints. Unit tests
with a fake `CustomObjectsApi`. Verified on `open-rl-dra` with a dev
api-server image, re-running the getting-started steps.

**Phase 2: failures reach the client.** Fail futures on release, drain
queues, expired-session semantics, `try_again` detail, `NoCapacity`
timeout, the log lines (5.3, 5.4). Tests in `test_session_lifecycle.py`.
E2E: drop the port-forward for three minutes during `tiny_sft` and assert
the client gets the released-workers error within one TTL plus one reaper
interval.

**Phase 3: the reference.** Settings generator and its test, the skill,
the release bundle target and `skill_url`, the marketplace manifest,
`llms.txt`, links from the landing page, README and getting-started, the
`RELEASING.md` step, doctor in the client image (5.5, 5.6). Re-verify
every number in the skill on the release the phase ships in.

Phases 1 and 2 are server changes and ship in the next release; phase 3
can land in parallel since the skill depends only on the endpoints from
phase 1.

## 8. Testing

- Unit: each new endpoint against a fake Kubernetes client and the
  in-memory store; settings generator against `class Settings`; future
  failure on release; heartbeat after expiry.
- Compatibility: `tiny_sft`, `tiny_rl` and one cookbook recipe against the
  changed server with the pinned SDK and the newest SDK, confirming the
  added fields are ignored.
- E2E on `open-rl-dra`: the getting-started steps, the doctor, and the
  port-forward drop scenario.

## 9. Risks

- The SDK could one day validate `try_again` strictly. Mitigation: the
  compatibility run above, and the `openrl` object is the only addition.
- Status endpoints add Kubernetes reads on a path agents may poll.
  Mitigation: a short in-process cache (5 s) and no watches.
- Failing futures on release changes behaviour for clients that relied on
  waiting through a worker restart. No such path exists today: a released
  worker never comes back for the same session.
