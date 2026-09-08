# NPCMicro13M

NPCMicro13M is a compact, local-first language model for believable game
characters. It is ready to run as a merchant, guard, villager, quest-giver,
companion, monster, or any other NPC whose behavior is supplied through game
state.

This repository contains the final SFT-trained model, matching tokenizer, and
deployment runtime. It is an inference release: training scripts, datasets,
optimizer state, and experiment caches are intentionally excluded so the
download is focused on what a game developer needs to ship.

## Why use it for NPCs?

- **Small and local.** At 13.6M parameters, it is practical for CPU-only
  machines and lightweight game servers. No hosted model, account, API key, or
  per-message bill is required.
- **State-driven characters.** Put the changing facts in the request—name,
  profession, location, faction, quest, prices, inventory, or weather—instead
  of creating a separate model for every NPC.
- **Bounded generation.** Each turn receives only the NPC state and the current
  player message. There is no hidden, ever-growing transcript that can cause
  the character to continue an old conversation.
- **Short in-world replies.** The 128-token context and concise response
  contract keep memory, latency, and server work bounded.
- **Grounded serving layer.** The optional grounded response layer uses facts
  explicitly present in state to reduce identity, location, profession, and
  simple factual hallucinations.
- **Game-owned memory.** If a game needs memory, store structured facts in the
  game server and send only the relevant facts on the next turn.

Compared with a large general-purpose chat model, NPCMicro13M gives up broad
reasoning and world knowledge in exchange for much lower resource use,
predictable short responses, local deployment, and easier per-game control.
Compared with a rule-only dialogue system, it can respond to varied wording
without requiring a separate rule for every phrasing.

## Quick start

Use Python 3.10 or newer:

```text
python -m pip install -r requirements.txt
python deployment/uomind_infer.py --bundle . --state "Your name is Mira. You are a baker in Britain." --player "Who are you?"
```

Verify the download and checkpoint:

```text
python deployment/uomind_infer.py --bundle . --verify-only
```

On Windows, double-click:

```text
deployment/START_UO_MIND_GUI.bat
```

The GUI lets you enter the NPC state once and hold a conversation. The visible
transcript is for the operator; each model turn remains single-pass and sends
only the fixed state plus the current player message.

## Python API

Load the model once and reuse the runtime for many NPC requests:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("deployment").resolve()))
from uomind_api import UOMindRuntime

runtime = UOMindRuntime(bundle=".", device="cpu")
reply = runtime.respond(
    "Your name is Sella. You are a ferrymaster in Dunmar. The fare is 3 silver.",
    "How much to cross?",
)
print(reply["text"])
```

The response includes both the grounded text and the raw model text for
development inspection. Use `raw_model=True` when you need to bypass the
grounded response layer for testing.

The API is intentionally one-turn. The caller owns conversation memory and
should pass relevant structured facts through `state` rather than sending an
unbounded transcript.

## Game integration pattern

Every request follows this simple contract:

```text
NPC STATE: fixed facts for this turn
PLAYER:    current player speech
NPC:       one concise answer
```

Example state fields for any game:

```text
Your name is Sella. You are a ferrymaster in Dunmar.
You charge 3 silver to cross the river.
You know the north and east crossings, but not distant kingdoms.
You are patient but wary of armed travelers.
```

The same model can serve hundreds of NPCs. Keep the model loaded once, pass a
different state string for each character, and serialize access or use a small
worker pool when several players speak at the same time.

## Model design

The final SFT checkpoint has 13,640,064 parameters:

- 256-wide hidden state
- 16 transformer layers
- 8 attention heads
- 1,024-wide feed-forward layers
- 128-token context
- 4,096-token matching tokenizer
- tied input/output embeddings
- RMSNorm pre-norm architecture
- fixed multi-scale relational geometry instead of learned positional
  embeddings

The design separates three responsibilities:

1. The model learns language patterns and dialogue behavior.
2. The game supplies current facts through `state`.
3. The deployment layer applies optional grounding and output handling.

This makes character facts easy to change, keeps one model reusable across a
whole game world, and avoids paying the memory cost of a general-purpose model
for every NPC interaction.

## Included files

```text
model/npcmicro13m_sft.pt       Stripped final SFT inference checkpoint
model/checkpoint_info.json     Provenance, architecture, and hash metadata
tokenizer/tokenizer.json       Matching tokenizer
deployment/uomind_infer.py     CLI inference and integrity verification
deployment/uomind_api.py       Programmatic Python API
deployment/uomind_grounded_response.py  Optional grounding layer
deployment/uomind_gui.pyw      Desktop conversation GUI
deployment/START_UO_MIND_GUI.bat  Windows launcher
SHA256SUMS.txt                 Frozen-artifact integrity hashes
```

The `.pt` file contains only the model configuration and state dictionary. It
does not contain optimizer state, epoch counters, training metrics, or other
training-session data.

## Fine-tuning for a different game

This deployment bundle deliberately omits the training toolchain. To create a
game-specific derivative, use a compatible PyTorch trainer and this checkpoint
as the starting point. Keep the tokenizer and architecture unchanged, format
examples as state/player/assistant triples, and include preservation examples
for identity, concise replies, uncertainty, and answer relevance.

For the safest adaptation strategy, train only the final transformer blocks
first, keep the base checkpoint unchanged, and compare the candidate against a
held-out set before deployment. The model is most reliable when the game sends
the facts that must be exact instead of expecting the weights to memorize live
prices, quest progress, inventories, or locations.

## Limitations

NPCMicro13M is specialized for short responses. It is not a replacement for a
database, quest engine, rules engine, or large reasoning model. Give the game
facts explicitly when accuracy matters, and treat generated text as dialogue,
not as authoritative game state.

## License

This repository is distributed under the Apache License 2.0. Third-party
dependencies, source materials, and downstream datasets remain subject to
their own licenses.
