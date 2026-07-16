# Receptionist Voice Agent

A voice AI receptionist built on Pipecat, with Moss for sub-10ms semantic
retrieval over your business's FAQ/policy knowledge base. Forked from Moss's
[pipecat-quickstart](https://github.com/usemoss/moss-samples/tree/main/pipecat-moss/pipecat-quickstart).

## What's here

- `bot.py` — the voice pipeline: STT → Moss retrieval → LLM → TTS, with a
  receptionist system prompt (honest fallback instead of guessing on
  hours/pricing/policy)
- `create-index.py` — uploads the knowledge base to Moss. Replace the
  placeholder docs with your real hours, services, pricing, and policies —
  one fact per document, not one giant block of text
- `env.example` — copy to `.env` and fill in your API keys
- `pipecat_moss/` — vendored copy of the `pipecat-moss` PyPI package, patched
  for pipecat-ai 1.5 (upstream still imports frames removed in 1.5) and
  updated to the current `moss` SDK. Delete it and restore the PyPI dep once
  upstream catches up.
- `Dockerfile.dev` — local development image (see platform notes below)
- `Dockerfile`, `pcc-deploy.toml` — for deploying to Pipecat Cloud once local
  dev works

## Platform notes (this machine)

- Python is pinned to 3.12 in `.python-version` — the Moss client can't
  compute query embeddings on 3.14.
- The `moss` SDK's native wheels need glibc >= 2.38. Debian 12 has 2.36, so
  on this machine the bot runs inside Docker (`Dockerfile.dev`, based on
  Debian 13). On a host with glibc >= 2.38 (Debian 13+, Ubuntu 24.04+),
  plain `uv sync` + `uv run bot.py` works natively.
- The embedding model (~87 MB) is cached at `~/.cache/moss-models/moss-minilm/`
  after first use.

## Setup

1. Get API keys: [Moss](https://portal.usemoss.dev),
   [Deepgram](https://console.deepgram.com/signup),
   [OpenAI](https://auth.openai.com/create-account),
   [Cartesia](https://play.cartesia.ai/sign-up)

2. ```bash
   cp env.example .env
   # fill in your keys in .env
   ```

3. Build the dev image:

   ```bash
   docker build -f Dockerfile.dev -t receptionist-dev .
   ```

4. Edit `create-index.py` — swap in your real business info, then upload it:

   ```bash
   docker run --rm -it --network host --env-file .env receptionist-dev \
       uv run create-index.py
   ```

5. Run the bot locally:

   ```bash
   docker run --rm -it --network host --env-file .env receptionist-dev
   ```

   Open http://localhost:7860 and click **Connect** to talk to it.

   (On a host with glibc >= 2.38 you can skip Docker entirely:
   `uv sync && uv run bot.py`.)

## What's NOT wired in yet (your next build steps)

- **FAQ vs. booking split** — right now every turn goes through Moss
  retrieval. Booking/rescheduling should route to a calendar tool call
  instead, kept separate from the retrieval path so it stays deterministic.
- **Low-signal turn skip** — short turns ("yes", "okay", "repeat that")
  don't need a retrieval call at all; add a cheap check before
  `moss_service.query(...)` to skip it on those turns.
- **Similarity threshold fallback** — the system prompt tells the LLM to be
  honest when it doesn't know, but you'll want an explicit similarity-score
  check on the Moss results so low-confidence retrievals never reach the LLM
  framed as "here's the answer."
- **`alpha` tuning** — `MOSS_ALPHA` in `.env` controls the semantic/keyword
  blend (1.0 = pure semantic, 0.0 = pure keyword). Worth tuning once you have
  real call transcripts — exact terms like staff names or service names
  often do better with some keyword weight mixed in.
