# Ollama on Google Colab

`templates/colab.py` is a **compute provider**, not a Gemini API provider. It
reimplements the authentication, assignment and Jupyter connection used by
[Google's official Colab CLI](https://github.com/googlecolab/google-colab-cli).
It allocates a Colab VM, starts Ollama there, and returns OpenAI-compatible
responses through LLMProxy. The CLI does not need to be installed.

## Setup

1. In **Providers → Add from Template**, add **Colab · Ollama**.
2. In **Accounts**, sign in with each Google account you intend to use. Approve
   the Colab/email permissions and paste the authorization code shown on
   Google's Cloud SDK landing page into the existing callback field. This works
   on remote/headless hosts; no localhost callback server is started.
3. Use `cl/qwen3:0.6b` for a small, fast-starting local model. Add custom models
   with a public Ollama tag as `upstream_model`, for example `qwen3:8b`. Set the
   context window, output limit, vision and reasoning capabilities truthfully.
4. Send a normal request to LLMProxy:

   ```bash
   curl http://localhost:8000/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"cl/qwen3:0.6b","messages":[{"role":"user","content":"Hello"}],"stream":true}'
   ```

   Include your local LLMProxy API key when one is configured.

The first real request allocates a VM and downloads Ollama and the requested
model. Further requests reuse the VM, kernel connection, installed runtime and
downloaded weights. The default keeps the model loaded in VRAM for the VM's
lifetime. There is no eager VM allocation, network call, Google SDK import or
CLI subprocess at LLMProxy startup.

## Settings

All settings are ordinary provider variables in the dashboard.

| Setting | Default | Meaning |
|---|---|---|
| `split_across_accounts` | `false` | Keep the highest-priority available account warm. Other accounts remain failover candidates. Enable to distribute **separate requests** across independent account/VM replicas. This is not cross-VM model sharding. |
| `load_balancing_strategy` | `round_robin` | Used when splitting is enabled: round robin, least used, random or fallback. Fallback minimizes cold VM starts. |
| `gpu` | `T4` | CPU, T4, L4, G4, A100 or H100. Unsupported values are rejected; there is no silent upgrade/fallback to expensive hardware. Ollama does not support TPU. |
| `high_mem` | `false` | Request high-RAM shape when supported. L4 has only one shape. |
| `vm_idle_seconds` | `900` | Release an owned VM after 15 minutes without requests. `0`: release after each request. `-1`: disable the plugin idle limit. |
| `vm_max_lifetime_seconds` | `3600` | Rotate/release a VM after one hour once in-flight work finishes. `0`: no plugin maximum. |
| `max_active_vms` | `0` | An optional process-local cap. `0` allows one VM per used account. A positive cap rejects additional VM starts until capacity is released; it never evicts busy VMs. |
| `model_keep_alive_seconds` | `-1` | Retain the model in memory for the VM lifetime. `0` unloads after a request; positive values are seconds. This does **not** ping or extend the VM lifetime. |
| `preload_model` | `true` | Load the requested model once during setup, not on every inference. Only missing model weights are pulled. |
| `startup_timeout` | `900` | Total VM allocation, kernel startup, Ollama installation, model download and preload budget. |
| `timeout` | `300` | Inference deadline; waiting for the per-account execution lock has the same bounded timeout. |
| `request_timeout` | `30` | Control API/network inactivity timeout. |
| `assign_retries` | `2` | Recover transient allocation timeouts/502/503/504, reconciling the same notebook hash before another POST. Auth and quota failures are not retried internally. |
| `assign_retry_backoff` | `0.5` | Exponential delay after allocation failure only; no unconditional startup sleep. |
| `health_check_seconds` | `30` | Hot requests skip control-plane health checks until this interval passes. `0` checks every request. No idle pings. |
| `ollama_version` | `latest` | Current Ollama download or a pinned release version. Installed binaries are reused, not reinstalled on each request. |
| `models_directory` | `/content/.llmproxy-colab/models` | Model cache on the VM. Can point at storage you have already mounted. The plugin does not mount Drive automatically. |
| `release_vms_on_shutdown` | `true` | Release owned VMs on graceful proxy shutdown or plugin deletion. Disable to reconnect after a proxy restart. Hot reload always preserves VMs. |
| `reasoning_effort` | `default` | Model default, or a supported reasoning level. Qwen3's `none` disables thinking and `high` enables it. |
| `account_switch_cooldown` | `60` | Router-owned account cooldown after quota/rate-limit failure. |
| `account_switch_wait` | `0` | No account-switch delay by default. |

Per-account VM settings can override provider defaults in
`account.extra_config.colab_settings`, for example:

```json
{"colab_settings":{"gpu":"L4","vm_idle_seconds":1800}}
```

Credentials remain in the account, never in provider variables. Account and
provider proxy settings are supported for both HTTPS control calls and the
Jupyter connection. The resolved router proxy takes precedence.

## Startup and resource safety

- Each used account gets **one independent owned VM**, and requests on that VM
  are serialized. Separate accounts can execute concurrently.
- Warm requests need no allocation, model download or CLI authentication.
- Ollama binds to loopback on the VM. Inference is relayed through authenticated
  Jupyter output; there is no public ngrok/Cloudflare server or extra tunnel key.
- Installation uses the official Ollama binary archive and only installs `zstd`
  if absent. It does not install a Python inference stack or NVIDIA drivers.
- Request/model data is JSON, not an executable shell command. Google tokens,
  external proxy secrets and account credentials are not sent into Ollama.
- Allocation state is saved **before** the POST, allowing recovery of a VM whose
  allocation succeeded before a timeout or process exit. Runtime-proxy tokens
  are refreshed from the assignment listing before expiry.
- Timers release VMs even when no later request arrives. Failed releases retain
  the owned endpoint and retry rather than forgetting a possibly billable VM.
- Reload/startup restores ownership and re-arms these timers without allocating
  new VMs. A replacement waits for the previous instance's active work before
  reconnecting to the same kernel.
- Idle/lifetime cleanup never interrupts a request. Closing a client stream
  interrupts the corresponding kernel execution and frees its request lock.
- Only assignments carrying this plugin's saved ownership marker are touched.
  Existing notebook/CLI sessions on the same Google account are not adopted,
  restarted or terminated.

Colab can reclaim VMs earlier due to inactivity, capacity or its own lifetime
limits. The plugin's settings cannot override these limits. There are no dummy
kernel executions or artificial keep-alive pings. Use this integration only with
a Colab plan and workload permitted by
[Colab's terms](https://research.google.com/colaboratory/tos_v6.html) and
[usage restrictions](https://research.google.com/colaboratory/faq.html).
Free-tier Colab restricts some remote/service workloads; account/GPU availability
is not guaranteed. Account splitting is not a quota bypass.

## Authentication and local import

Sign-in uses the same public installed-application OAuth client and registered
remote landing page as Google's CLI, with PKCE and state checking. That public
client configuration is not a Google account secret. Account access/refresh
tokens are persisted in LLMProxy's private runtime data, never in source.

**Import local** supports authorized-user JSON credentials from:

1. `GOOGLE_APPLICATION_CREDENTIALS`, when set;
2. `~/.config/gcloud/application_default_credentials.json`;
3. the legacy `~/.config/colab-cli/token.json`.

It validates actual Colab/email scopes after refreshing. To create user ADC:

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

The current CLI stores OAuth in an OS keyring. This standalone plugin does not
open the keyring or trigger unlock prompts; use the dashboard sign-in or user
ADC instead. Local import never alters/deletes the source credential file and
does not support service accounts or metadata-server credentials.

## Model capabilities and telemetry

- Tools, tool results, structured JSON output, sampling options and inline
  images are translated to Ollama's native `/api/chat` shape. Remote images are
  materialized with the shared LLMProxy multimodal helper.
- Native thinking controls are discovered with `/api/show`. Thinking output is
  returned as `reasoning_content`. GPT-OSS supports `low/medium/high` rather than
  disabling thinking; add only the true levels to custom model metadata.
- Ollama does not support forced/named tool choice, `n > 1`, or enforcing
  `parallel_tool_calls=false`. Those requests fail explicitly instead of
  silently ignoring the client's constraint. `tool_choice=none` removes tools.
- Prompt and completion counts come from `prompt_eval_count` and `eval_count`.
  Recent Ollama versions also supply `prompt_eval_cached_count`; it reaches
  request logs and the dashboard cache hit rate. Older versions leave cache
  counters unknown rather than inventing hits.
- Ollama's cache is implicit, VM/model-local KV caching. Native extension hints
  are retained, but Ollama does not implement durable `cachedContent` or
  Anthropic cache breakpoints. It does not expose a separate reasoning-token
  counter. Warm primary-account routing improves the chance of a prefix hit.
- Streaming terminal usage is collected even when the client does not opt in.
  The router exposes usage chunks only when requested, as for other providers.
- The account usage meter reads real compute-unit balance, hourly consumption
  and active assignment count from Colab's `/tun/m/ccu-info`; no quota bars are
  fabricated.
- Cloud Ollama tags are rejected. This provider runs the model on **your Colab
  VM**, not Ollama Cloud or Google's Gemini API.

## Replacing the broken Gemini-as-Colab adapter

This checkout ships just `templates/colab.py`, not a second `google_colab`
template. In an installation with the old plugin:

1. Stop sending traffic to the old Colab provider.
2. Delete the old `colab` / `google_colab` provider from the dashboard and add
   **Colab · Ollama** from the template gallery.
3. Sign in again. The previous OAuth scopes did not include Colab, and its Gemini
   base URL cannot allocate VMs. Old credentials/model aliases are not silently
   converted into working Colab credentials.
4. Replace old Gemini model aliases with an actual local Ollama tag.

No runtime plugin files, Google accounts or other worktrees are removed by a
repository update. Real account, OAuth, allocation and inference failures are
never replaced by simulated successes. Simulation exists only for the synthetic
no-account playground and explicitly named test/mock credentials.

## Validation

```bash
python -m py_compile templates/colab.py
python skills/llmproxy-plugin/scripts/validate_plugin.py templates/colab.py --smoke
pytest -q tests/test_colab.py
pytest -q --ignore=tests/test_codex_and_example_response.py
```

Tests mock the Google control plane, Jupyter WebSocket protocol, kernel output,
Ollama responses, OAuth and timers. Passing mocks does **not** establish current
live Colab backend compatibility or actual cold-start timing; a Google account
and a permitted real VM allocation are required for that verification.
