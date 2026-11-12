# Agent Skills

This directory contains vendor-neutral skills for AI coding agents working on
LLMProxy. Each skill is a directory with a `SKILL.md` entrypoint and optional
`references/`, `assets/`, and `scripts/` folders.

## Available skills

| Skill | Purpose |
|---|---|
| `llmproxy-plugin` | Build, review, or repair provider plugins: metadata, models, variables, accounts, streaming, tool calls, usage, proxies, tests. |

## Use a skill

Point the external agent at the `SKILL.md` file:

```text
Read skills/llmproxy-plugin/SKILL.md and use it to add a provider plugin for <API>.
```

If the agent only discovers skills from a fixed directory, copy or symlink the
skill directory into that location. Common conventions:

```bash
# Claude Code
mkdir -p .claude/skills
ln -s ../../skills/llmproxy-plugin .claude/skills/llmproxy-plugin

# OpenCode / generic agent-skill layouts
mkdir -p .agents/skills
ln -s ../../skills/llmproxy-plugin .agents/skills/llmproxy-plugin
```

Do not commit local symlinks unless the repository decides to standardize on that
agent layout.

## Validate a plugin

```bash
python skills/llmproxy-plugin/scripts/validate_plugin.py templates/my_provider.py
python skills/llmproxy-plugin/scripts/validate_plugin.py templates/my_provider.py --smoke
```

The first command loads the plugin and checks metadata/signatures without network
access. `--smoke` also runs simulation-mode streaming and non-streaming requests.

## Adding a new skill

Create a new directory under `skills/`:

```text
skills/
  my-skill/
    SKILL.md
    references/
    assets/
    scripts/
```

Use YAML frontmatter with a unique `name` and a specific `description` that tells
the agent when to load the skill. Keep the main `SKILL.md` focused on workflow and
put long field references in `references/`.
