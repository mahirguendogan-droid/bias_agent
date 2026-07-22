# Deploying to Hugging Face Spaces

The app is Gradio-based, so it runs on a free Hugging Face Space with no code changes.
Everything except the three steps below is already wired up.

## One-time setup

**1. Create the Space**

Go to [huggingface.co/new-space](https://huggingface.co/new-space):

| Field | Value |
|---|---|
| Owner | your HF username |
| Space name | `AutoBiasAgent` |
| License | MIT |
| SDK | **Gradio** |
| Hardware | CPU basic (free) |

**2. Give the Space an API key**

In the Space → **Settings** → **Variables and secrets** → **New secret**:

| Name | Value |
|---|---|
| `OPENAI_API_KEY` | your OpenAI key |

The agent reads it via `os.environ.get("OPENAI_API_KEY")`. Without it, the statistical
phases still run — only the LLM explanation and judge phases fail.

> Add the key as a **secret**, never as a public variable, and never commit it to the repo.

**3. Wire up automatic deploys from GitHub**

In this GitHub repo → **Settings** → **Secrets and variables** → **Actions**:

| Type | Name | Value |
|---|---|---|
| Secret | `HF_TOKEN` | a **write** token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| Variable | `HF_SPACE` | `<your-username>/AutoBiasAgent` |

From then on, every push to `main` mirrors the repo to the Space via
[`.github/workflows/sync-to-hf.yml`](.github/workflows/sync-to-hf.yml). The Space front matter
already sits at the top of `README.md`, so the push needs no file rewriting.

Until `HF_SPACE` is set the deploy job is skipped, so nothing breaks in the meantime.

## Manual deploy (no GitHub Actions)

```bash
git remote add space https://huggingface.co/spaces/<your-username>/AutoBiasAgent
git push --force space main
```

## Cost

A full audit of the bundled Titanic dataset is 3–5 `gpt-4o-mini` calls, roughly
**$0.0005**. The Space itself is free on CPU basic.

## Notes

- `sdk_version` in the README front matter is pinned to **6.20.0**, the version the UI is
  tested against. Gradio 6 moved `css` off the `Blocks` constructor, so an unpinned SDK can
  silently change the layout.
- Cold start takes ~30 s on free CPU hardware while the Space installs dependencies.
