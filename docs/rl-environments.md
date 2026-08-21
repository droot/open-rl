# OpenRL and RL Environments

*Where OpenRL stops, where RL environments begin, and how sandbox projects fit in between.*

A lot of people have asked the same two questions recently: how does OpenRL interact with RL
environments, and is there an opportunity to integrate with agent-sandbox or agent substrate? This
document answers both at a high level. A companion design doc covers the implementation detail.

---

## Where OpenRL draws its boundary

An RL workload has many moving parts. Two of them — training and inference — are needed by *every*
RL run, no matter the domain. The rest are not.

RL environments and reward graders depend entirely on what you are trying to teach the model. We
have seen environments that are a single Python function, and environments that are a full isolated
container per trajectory. We have seen graders that are a regular expression, graders that execute
code, and graders that are another language model applying a rubric. There is no useful common
abstraction across that range that we could put behind an API without getting in someone's way.

So OpenRL's API wraps training and sampling, and stops there. Everything above it — the algorithm,
the environment, the reward — stays in the researcher's code, where it can change freely.

**This is a deliberate boundary, not an unfinished roadmap.** It is the same reasoning behind the
README's "OpenRL is not an RL framework."

The honest consequence is that it leaves a real gap for domains with complex environments. That gap
is where the sandbox projects come in. Our job is not to close it with our own abstraction — it is
to show people how the pieces fit together, and to make sure the Kubernetes-shaped piece exists.

---

## What a rollout actually looks like

Before the integration question makes sense, it helps to be precise about what an RL training step
generates.

Each training step produces a batch of **trajectories**. Generating them is called *sampling*, or
*rollout*. A single trajectory is:

- an initial prompt;
- some number of **turns**, each one a tool call the model asked for and the result of actually
  running it;
- a final answer;
- one or more reward scores.

That multi-turn loop is the *agentic loop*, and whatever code drives it — deciding when to call a
tool, feeding the result back, deciding when to stop — is the **agent harness**.

The whole integration question reduces to one thing: **who owns the harness?** That determines who
holds the connection to the model, where the trajectory is stored, and which version of the weights
each turn was generated from.

---

## Four patterns

```
P1  loop ── tool()                          no sandbox
P2  loop ── harness ──→ [ sandbox: tools ]  harness outside
P3  loop ──────────────→ [ sandbox: harness + tools ]  harness inside
P4  loop ──────────────→ shared env service
```

**P1 — inline.** The tool call is just a function in the training loop. Our text-to-SQL example works
this way: generate SQL, run it against a local database, score the result. Perfect for verifiable,
self-contained tasks. Falls apart the moment the model needs to run untrusted code, keep state, or
touch a network.

**P2 — sandboxed tools.** The training loop still runs the agent loop, but each tool call is executed
inside an isolated sandbox. The sandbox is a safe pair of hands, not a participant.

This is the mainstream pattern. It is what most real agentic RL does today, it is what our own
Harvey LAB recipe does, and it works against OpenRL right now with no changes to anything.

**P3 — sandboxed harness.** A real agent — Claude Code, ADK, something speaking MCP — runs *inside*
the sandbox, pointed at the model. It works the task on its own and reports back when finished.

You want this when the fine-tuned model has to work well in a harness you don't control. In P2 you
are training against your own reimplementation of an agent loop and hoping the behaviour transfers.
In P3 you train against the real thing. This is the pattern that needs work from us.

**P4 — environment as a service.** A shared long-lived environment service instead of per-trajectory
sandboxes. Worth knowing about; not what this document is about.

| | P1 | P2 | P3 | P4 |
| --- | --- | --- | --- | --- |
| Who runs the agent loop | training loop | training loop | the sandbox | env service |
| Isolation | none | per trajectory | per trajectory | shared |
| Trains against a real harness | no | no | **yes** | partly |
| Works on OpenRL today | **yes** | **yes** | not yet | yes |

---

## Why we are not building an environment API

Because a good one already exists and we get it for free.

OpenRL implements Tinker-compatible APIs, which means the tinker-cookbook works against it
unmodified. The cookbook already provides the environment abstraction, the multi-turn agent loop,
group-based rollouts for GRPO, and — the part people miss — **a pluggable sandbox interface with
explicit support for custom backends.**

It also already solves a set of problems we would otherwise have rediscovered the hard way: what to
do when a rollout crashes halfway through, how to bound turns and wall-clock time, how to keep one
oversized trajectory from poisoning its group, and how to make failures visible in the metrics
instead of silently absent. These are not small details. They are most of what makes agentic RL
annoying, and they are already handled.

Two consequences follow. First, we should not define our own environment, tool, or sandbox
abstraction — we would be building a worse version of something our users already have. Second, our
actual contribution is narrow and obvious: **the cookbook's sandbox interface ships backends for
Modal and SandboxFusion. There is no Kubernetes backend.** That is the gap, and it is a few hundred
lines, not a subsystem.

There is one thing the cookbook does *not* cover, and it is exactly P3. Its model assumes the
training loop drives each turn. When the harness lives inside the sandbox, nobody is driving — the
agent has its own connection to the model. That is why P2 needs nothing from us and P3 needs real
work.

---

## agent-sandbox and agent substrate are not alternatives

The question usually arrives as "agent-sandbox *or* substrate?" They solve different problems and
the answer is likely both, at different times.

**agent-sandbox** gives you an isolated, individually addressable sandbox with a real lifecycle —
create, pause, resume, delete — and pre-warmed pools so you are not paying container startup on every
claim. It is about **isolation**.

**agent substrate** packs many logical agents onto far fewer running workers, suspending and resuming
them in under a second. Its founding observation is that agent workloads spend most of their life
idle. It is about **density**.

Which one you need depends on the shape of your rollouts, and here we were initially wrong about the
shape.

The intuition is that RL means enormous churn — thousands of sandboxes created and destroyed every
training step — which would make claim latency the number that matters. Two real recipes say
otherwise. Both the cookbook's Terminal-Bench recipe and our own Harvey LAB recipe run a few dozen
concurrent sandboxes, each living for a long time — up to an hour — and spending nearly all of that
time blocked waiting on the model to generate the next turn.

That is not a churn problem. **It is a few dozen long-lived, mostly-idle containers**, and it is
substrate's founding observation almost word for word. Two unrelated recipes landing on the same
profile is worth taking seriously.

So: **agent-sandbox is the right first backend** — simpler, stable identity, and it lines up closely
with the interface the cookbook already expects. **Substrate is the answer when we scale**, and more
so for P3 than P2, because P3 multiplies the number of idle agents.

Neither project has RL-specific APIs, and neither should. The piece neither supplies is the
trajectory contract for P3 — how a sandboxed agent records what it did and tells the training loop it
has finished. That one is ours.

---

## We already have a working example

This is not hypothetical. Our Harvey LAB recipe does live agentic RL on a legal benchmark: the model
works real tasks in a container with document, shell, and file tools; a judge grades the rubric; the
pass fraction is the reward. Held-out criterion pass rate went from 48.7% to 67.6% over twenty
training steps.

It is a complete, validated P2 implementation. The harness runs in the training loop, tool calls are
executed inside per-rollout containers, and the containers are torn down at the end of each group.

It also has exactly one problem, and it is the interesting one. The containers run under podman on
the same machine as the trainer, and the grader reads files off that machine's local disk. That works
beautifully on a single 8-GPU box and cannot leave it. Roughly fifty containers compete with training
for the same CPU, memory, and disk, and there is no way to add more.

**Breaking that coupling is the port and the payoff at the same time.** It is the concrete argument
for the whole integration: same recipe, same reward curve, sandboxes moved off the trainer and onto
the cluster, with room to scale.

---

## What we propose

**First, prove P2 on Kubernetes.** Swap the local container backend in the Harvey LAB recipe for an
agent-sandbox-backed one and reproduce the existing reward curve with the sandboxes running
elsewhere on the cluster. Same recipe, same numbers, no longer stuck on one machine. We have a
published baseline to measure against, which makes this unusually easy to judge.

The real work is not the sandbox API. It is untangling the assumption that the environment and the
grader share a filesystem.

**Second, contribute the backend upstream if we can.** Written as a cookbook sandbox backend rather
than something LAB-specific, the same work gives a Kubernetes option to every existing sandbox recipe
— Terminal-Bench, SWE-bench, competitive programming — and benefits Tinker users generally, not only
OpenRL users. That is a better trade than keeping it in our examples directory, though it means an
external review cycle.

**Third, and only then, P3.** Harvey LAB is the best candidate we have, because the benchmark ships
its own agent harness — so a model tuned for it will be deployed in that harness, not in ours. Right
now we train against a reimplementation and hope. P3 closes that gap by construction.

P3 needs three things from OpenRL that do not exist yet: a way for a standard agent to talk to our
sampler without corrupting the token stream, an agreed format for a sandboxed agent to report its
trajectory, and a story for what happens when a rollout outlives the weights it started with. Those
are real, and they are ours to build — but none of them block the first two steps.

---

## The one thing worth understanding underneath all this

OpenRL passes tokens end to end. When the training loop asks for a sample it sends token IDs, and
what comes back is token IDs and their log-probabilities — never text that has to be re-tokenized on
the way back in.

That matters more than it sounds. Re-tokenizing between sampling and training is a classic source of
silent mismatch between what the model generated and what the trainer thinks it generated, and it
destabilizes runs in ways that are miserable to debug. OpenRL avoids it structurally today, and the
cookbook's environment abstraction is token-level too, so P1 and P2 preserve the property for free.

P3 is the pattern that puts it at risk, because real agent harnesses speak text over an
OpenAI-compatible API. Bridging that gap *without* reintroducing re-tokenization is the single most
important technical decision in this whole area — and it is the one open question we most need to
settle.

---

## Open questions

1. Do we take on an OpenAI-compatible sampling endpoint? This is the only thing blocking P3, and it
   has to be done in a way that preserves exact tokens.
2. Does the Kubernetes sandbox backend go upstream into tinker-cookbook, or stay in our examples?
3. Sandbox images are per-task in these benchmarks, which complicates pre-warmed pools. Prebuild and
   cache, or accept cold starts?
4. When a rollout outlives the training step that started it, do we let it continue on older weights
   by default, or make that opt-in?
5. Substrate's control plane is Go and gRPC; RL loops are Python. Is there a client path?
