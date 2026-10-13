---
title: "Point your coding agent at OpenRL"
date: 2026-10-12
authors: ["Sunil Arora"]
summary: "OpenRL vVERSION describes itself to the agent that drives it: the server reports its version and limits and publishes every setting in its OpenAPI document, one endpoint says what a run is waiting for, and a skill tells the agent what differs from hosted Tinker. Install it in one command."
tags: ["release", "agents"]
draft: true
---

Most people who train with OpenRL do not type the training loop themselves. They tell a coding agent what they want, and the agent writes the loop, runs it, and watches it. This release is for that agent.

## The problem

An agent cannot ask a colleague. When we tested the v0.0.6 getting-started guide with an agent at the keyboard, a run stopped with no error. To find out why, the agent read the server source, ran `kubectl`, and finally read Redis inside the cluster. It found the answer. It should not have had to look.

Three facts were missing from where the agent was looking: which server it was talking to and what that server could do, what the cluster was doing while a call blocked, and which of its Tinker habits do not apply on a self-hosted server.

## What is new

<svg viewBox="0 0 760 230" width="760" role="img" aria-label="An agent talks to the OpenRL API server. The server now answers three questions from the endpoint alone: who are you, what is my run waiting for, and how do I use you. The cluster stays behind the server." style="max-width:100%;height:auto;font-family:inherit;font-size:14px">
  <defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5 0 10z" fill="currentColor"/></marker></defs>
  <g fill="none" stroke="currentColor" stroke-width="1.5">
    <rect x="20" y="70" width="150" height="90" rx="8"/>
    <rect x="300" y="20" width="200" height="190" rx="8"/>
    <rect x="610" y="70" width="130" height="90" rx="8"/>
    <line x1="170" y1="115" x2="300" y2="115" marker-end="url(#a)"/>
    <line x1="500" y1="115" x2="610" y2="115" marker-end="url(#a)"/>
  </g>
  <g fill="currentColor">
    <text x="95" y="110" text-anchor="middle" font-weight="600">Your agent</text>
    <text x="95" y="132" text-anchor="middle" opacity=".7">on your laptop</text>
    <text x="400" y="48" text-anchor="middle" font-weight="600">OpenRL API server</text>
    <text x="400" y="88" text-anchor="middle">who are you?</text>
    <text x="400" y="106" text-anchor="middle" opacity=".7" font-size="12">get_server_capabilities</text>
    <text x="400" y="138" text-anchor="middle">what is my run waiting for?</text>
    <text x="400" y="156" text-anchor="middle" opacity=".7" font-size="12">/api/v1/openrl/models/{id}</text>
    <text x="400" y="188" text-anchor="middle">how do I use you?</text>
    <text x="400" y="204" text-anchor="middle" opacity=".7" font-size="12">skill_url</text>
    <text x="675" y="110" text-anchor="middle" font-weight="600">Cluster</text>
    <text x="675" y="132" text-anchor="middle" opacity=".7">no kubectl needed</text>
    <text x="235" y="105" text-anchor="middle" opacity=".7" font-size="12">Tinker API</text>
    <text x="555" y="105" text-anchor="middle" opacity=".7" font-size="12">Workloads</text>
  </g>
</svg>

**The server describes itself.** `get_server_capabilities` has a new `openrl` object. It names the release, how the server runs workers, the deployment limits, and where the settings are described: every `openrl.*` setting, with its type, default and description, is published once, as `components.schemas.OpenRLSettings` in `/openapi.json`, and rendered at `/docs`. It is pydantic's own schema for the settings class, so it cannot be stale, and nothing else carries a copy.

**One endpoint says what a run is waiting for.** `GET /api/v1/openrl/models/{model_id}` returns the model's settings, whether its session is live, and its workers. Each worker has a `state` from a fixed set of five, the same five on every deployment:

| `state` | Meaning |
|---|---|
| `waiting_for_placement` | no accelerator assigned yet |
| `starting` | placed; pulling the image, mounting volumes, or loading the model |
| `ready` | serving requests |
| `failed` | exited with an error, or the request cannot be satisfied |
| `stopped` | exited normally, or being released |

An agent branches on `state`. The scheduler's own words go in a free-text `detail` for a human to read. A one-line `summary` says what to know first.

**A skill for the OpenRL bits only.** `skills/openrl/` ships with the release. It covers connecting, the settings, what the cluster is doing while you wait, and OpenRL's own errors. How to train is not in it. The [tinker-cookbook skills](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/skills) cover that far better than we could, so the OpenRL skill names them, says which to load for which question, and carries one table of where their hosted-service advice does not apply on a self-hosted server.

## See it

Ask the server who it is:

```bash
export OPENRL=http://127.0.0.1:9003   # your port-forward
curl -s $OPENRL/api/v1/get_server_capabilities | jq .openrl
```

```json
{
  "version": "vVERSION",
  "worker_launcher": "scheduler",
  "fft_enabled": true,
  "settings_schema_url": "/openapi.json#/components/schemas/OpenRLSettings",
  "limits": {"session_ttl_seconds": 120, "session_reap_interval_seconds": 30},
  "skill_url": "https://github.com/gke-labs/open-rl/releases/download/vVERSION/openrl-skill.tar.gz"
}
```

Ask it what a run is waiting for. This is a real response from a LoRA run on `google/gemma-4-e2b`, two minutes in. The trainer is up. The sampler's pod is running, but vLLM is still loading the model, so the sampler is `starting`, not `ready`:

```bash
curl -s $OPENRL/api/v1/openrl/models/fd51fb5c-f0f4-475a-a986-eb146826309f \
  | jq '{session_live, workers, summary}'
```

```json
{
  "session_live": true,
  "workers": [
    {"role": "sampler", "state": "starting",
     "detail": "Running on gke-open-rl-dra-l4-sampler-2x-1ad5171e-bs96, loading the model",
     "age_seconds": 120},
    {"role": "trainer", "state": "ready",
     "detail": "Running on gke-open-rl-dra-l4-trainer-2x-67d40c4c-sr2c",
     "age_seconds": 202}
  ],
  "summary": "sampler starting: Running on gke-...-bs96, loading the model, 120s;
              trainer ready: Running on gke-...-sr2c, 202s"
}
```

That is the usage model: launch the job, then ask the server about the model from another shell while a call blocks. The examples in the repository print the model id and that `curl` right after `create_model`, so a run that is waiting looks like a run that is waiting, not like a hang:

```
[tiny-sft] model_id=fd51fb5c-f0f4-475a-a986-eb146826309f
[tiny-sft] status: curl -s http://127.0.0.1:9003/api/v1/openrl/models/fd51fb5c-f0f4-475a-a986-eb146826309f | jq .summary
```

After the client exits, the same endpoint says that the session expired, and then that the workers were released. The agent no longer needs the cluster to learn either fact.

## Install the skills for your agent

Two skills, one command each. The OpenRL skill is a tarball on the release; the server tells you which one matches it. The cookbook skills are a directory in their repository.

Most agents share one skills directory, `~/.agents/skills/`. [Codex](https://developers.openai.com/codex/skills/), [Gemini CLI](https://geminicli.com/docs/cli/skills/), [Cursor](https://cursor.com/docs/context/skills), [GitHub Copilot](https://docs.github.com/en/copilot/concepts/agents/about-agent-skills), [VS Code](https://code.visualstudio.com/docs/copilot/customization/agent-skills) and [OpenCode](https://opencode.ai/docs/skills/) all read it. Inside a repository, the same layout under `.agents/skills/` scopes the skills to that project.

```bash
mkdir -p ~/.agents/skills

# The OpenRL skill that matches your server
SKILL_URL=$(curl -s $OPENRL/api/v1/get_server_capabilities | jq -r .openrl.skill_url)
curl -sL "$SKILL_URL" | tar -xz -C ~/.agents/skills

# The tinker-cookbook skills, for how to train
git clone --depth 1 https://github.com/thinking-machines-lab/tinker-cookbook /tmp/tinker-cookbook
cp -R /tmp/tinker-cookbook/skills/{research,debug} ~/.agents/skills/
```

[Claude Code](https://code.claude.com/docs/en/skills) reads `~/.claude/skills/` instead, so replace the directory in the commands above. Both repositories also publish a plugin marketplace, which keeps the skills updated for you:

```
/plugin marketplace add gke-labs/open-rl
/plugin install openrl@open-rl
/plugin marketplace add thinking-machines-lab/tinker-cookbook
/plugin install tinker@tinker-skills
```

An agent with no skills directory can still be pointed at the two `SKILL.md` files. Each one links its references by relative path.

## Then ask

With the skills loaded, these prompts work as you would hope:

- "Check the OpenRL server at `$OPENRL` and tell me what it can do." The agent reads capabilities and the OpenAPI document and reports the release, the launcher mode and the settings.
- "Fine-tune `google/gemma-4-e2b` on this dataset with LoRA." The agent uses the cookbook's `research` skill for the loop and the OpenRL skill for the connection and settings.
- "The run has printed nothing for five minutes. Is it hung?" The agent reads the model view, compares the worker ages with the normal durations in the skill, and answers from evidence.
- "Switch this to full fine-tuning on an exclusive GPU." The agent sets `openrl.fine_tuning_type=full` and `openrl.exclusive=true` as settings, not code.

## What is not there yet

Two things are known and planned. First, when the server releases a run's workers, the client still learns about it by polling forever; the next release fails the pending call with the reason. Second, the model view is scoped to one model id on purpose: a fleet-wide view needs a decision about what one user may see of another's work, and OpenRL does not have a tenancy model yet. Both are written up in [design 016](https://github.com/gke-labs/open-rl/blob/main/docs/designs/016-agent-friendly-openrl.md).

## Recap

Point your agent at the endpoint. It learns what the server is from `get_server_capabilities`, what a run is waiting for from the model view, and how to use OpenRL from the skill the server links. Install the OpenRL skill and the tinker-cookbook skills side by side, in `~/.agents/skills/` or your agent's own directory, and ask.

The release notes are on [GitHub](https://github.com/gke-labs/open-rl/releases/tag/vVERSION). The [getting-started guide](https://github.com/gke-labs/open-rl/blob/main/docs/getting-started.md) has the deployment side.
