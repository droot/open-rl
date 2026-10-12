# openrl.* settings

Generated from `class Settings` in `src/server/api_server.py` by
`scripts/gen_settings_doc.py`; do not edit by hand. A running server reports
the same table under `openrl.settings` in `GET /api/v1/get_server_capabilities`.

Pass settings as `user_metadata` on `create_session` or `create_model`, or as
`TINKER_TAGS=openrl.key=value,openrl.other=value` in the client's environment.
Model settings override session settings. Values are strings on the wire
(`"true"`, `"8"`); an unknown `openrl.` key or a bad value is a 400 that names
the setting.

```python
client = tinker.ServiceClient(base_url=..., user_metadata={"openrl.fine_tuning_type": "full"})
```

| Setting | Type | Default | Values | Description |
|---|---|---|---|---|
| `openrl.sampler_replicas` | int | `1` | 1 to 8 | Single-GPU samplers that drain the model's sampling queue together. |
| `openrl.exclusive` | bool | false |  | A workload no other workload shares. The model gets its own trainer and sampler, and nothing time-slices their GPUs. |
| `openrl.trainer_backend` | string | `pytorch` |  | The trainer the model runs on: pytorch, automodel (LoRA only), or an image like ghcr.io/org/trainer:tag that runs its own trainer. |
| `openrl.trainer_gpus` | int | `1` | 1 to 8 | GPUs the trainer drives as one data-parallel torchrun group. More than one is Automodel only and makes the model exclusive. |
| `openrl.trainer_cp` | int | `1` | 1 to 8 | Trainer GPUs that share one sequence, as context parallelism. The rest of trainer_gpus is data parallel. |
| `openrl.fine_tuning_type` | string | unset | `lora`, `full` | lora unless set. A model restored from a checkpoint takes the checkpoint's type. |
| `openrl.weight_sync` | string | unset | `delta`, `full` | Full fine-tuning only. How the trainer hands each step's weights to the sampler: delta (changed weights only, the default) or full. |
| `openrl.fft_seed` | int | unset |  | Full fine-tuning only. The trainer's random seed. |
| `openrl.fft_cpu_offload` | bool | unset |  | Full fine-tuning only. Park trainer state in host memory between turns. Exclusive models default to off. |
