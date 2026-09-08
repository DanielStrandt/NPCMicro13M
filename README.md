# NPCMicro13M

NPCMicro13M is a small, local-first language model for giving characters in
games a believable voice. Fine-tune it for a merchant, guard, villager,
quest-giver, companion, monster, or any other NPC in almost any game world.

You provide the facts that matter—name, role, location, knowledge, attitude,
and current world state—then teach the model the final behavior and vocabulary
you want through ordinary JSONL examples. The included runtime can then answer
one player message at a time using only the NPC state and the current message.

## Why this design is useful

NPCMicro13M is designed for game servers and embedded applications rather than
for open-ended assistant chat.

- **Small enough to run locally.** The 13.6M-parameter model is practical to
  test and fine-tune on modest hardware, and it can run on CPU-only servers.
- **Game state stays outside the weights.** Put changing facts in the state
  supplied by the game—name, profession, town, quest, prices, faction, or
  weather—instead of retraining for every NPC.
- **One-turn generation is predictable.** Each request contains the NPC state
  and the current player speech. The model does not accidentally continue a
  hidden transcript or consume an ever-growing conversation history.
- **Short replies are a strength.** The 128-token context is focused on
  concise in-world dialogue, which keeps latency, memory, and server load low.
- **Fine-tuning is targeted.** The trainer can update only the final
  transformer blocks first, reducing the chance that a small custom dataset
  erases the model's basic conversational behavior.
- **Deployment is yours.** There is no hosted endpoint, account, API key, or
  per-message charge required. Use the Python API, HTTP API, command line, or
  desktop GUI.

This is a good fit when a large general-purpose model would be too expensive,
too slow, too dependent on a network service, or too difficult to keep inside
strict NPC boundaries. A large model remains the better choice for long-form
reasoning, broad factual research, or unrestricted multi-turn chat.

## What is included

- Trainer-compatible model checkpoint and matching 4,096-token tokenizer.
- Fine-tuning trainer with preservation-first evaluation and checkpoint
  promotion.
- Example custom-task dataset and manifest generator.
- The exact v9 repair/preservation corpus and training lineage.
- Local CLI inference, grounded response runtime, Python API, and double-click
  desktop GUI.
- Hash manifests, CI validation, and deployment documentation.

## Use it in three steps

### 1. Install and try the model

Use Python 3.10 or newer:

```text
python -m pip install -r requirements.txt
python deployment/uomind_infer.py --bundle . \
  --state "Your name is Mira. You are a baker in Britain." \
  --player "Who are you?"
```

On Windows, double-click `deployment/START_UO_MIND_GUI.bat` to open the
conversation GUI. For server integration, import `UOMindRuntime` as shown in
the Python integration section below.

The CLI uses the grounded response layer by default. Add `--raw-model` when
you want to inspect the model's unprocessed output during development.

### 2. Create examples for your game

Each training row is one complete state/question/answer example:

```json
{"messages":[
  {"role":"system","content":"Your name is Sella. You are a ferrymaster in Dunmar. You know the river crossings and charge 3 silver."},
  {"role":"user","content":"How much to cross?"},
  {"role":"assistant","content":"Three silver to cross, friend. The river runs swift today."}
],"meta":{"category":"commerce","family":"ferry_price","preservation":false}}
```

For another game, replace the state with that game's facts and write answers
in its voice. Useful categories include:

- identity, role, home, faction, and relationships;
- services, prices, items, quests, and local procedures;
- greetings, farewells, rumors, warnings, refusals, and uncertainty;
- setting-specific names, slang, lore, and terminology;
- out-of-scope questions where the NPC should admit it does not know.

Use several differently worded questions for each behavior. Include
preservation examples for behaviors you want to keep, especially identity
handling, concise replies, uncertainty, and answer relevance. A small,
balanced dataset is usually safer than many near-duplicate examples.

Create a manifest for your dataset:

```text
python training/make_manifest.py training/corpus/my_game.jsonl
```

### 3. Fine-tune and deploy your candidate

Start with a low-cost final-block update:

```text
python training/uomind_repair_finetune_v3.py \
  --bundle . \
  --checkpoint model/v9_base_finetune.pt \
  --allow-source-mismatch \
  --corpus-dir training/corpus \
  --corpus-glob my_game.jsonl \
  --require-manifests \
  --output-dir outputs/my_game \
  --train \
  --device cpu \
  --epochs 1 \
  --train-last-n-layers 4
```

Use `--device cuda` on a compatible NVIDIA machine. The preferred promoted
candidate is:

```text
outputs/my_game/checkpoints/best_preserved.pt
```

Keep the original checkpoint unchanged and give each experiment its own
output directory. Test the raw and grounded responses manually before putting
the candidate into a live game server.

## Integration model

The runtime contract is intentionally simple:

```text
NPC STATE: fixed facts for this turn
PLAYER:    the current player message
NPC:       one concise answer
```

The visible GUI transcript is for the human operator. It is not automatically
sent back as hidden context, so every turn is isolated and predictable. If a
game needs memory, the game server should store structured facts—quest stage,
relationship, inventory, prior promises—and include only the relevant facts
in the next state string.

Python integration:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("deployment").resolve()))
from uomind_api import UOMindRuntime

runtime = UOMindRuntime(bundle=".", checkpoint="model/v9_base_finetune.pt", device="cpu")
reply = runtime.respond(
    "Your name is Sella. You are a ferrymaster in Dunmar.",
    "Can you take me across?",
)
print(reply["text"])
```

The HTTP API exposes the same one-turn contract for game-server integration.
It binds to localhost by default; add authentication and access control before
exposing it to another machine.

## Model design

The included architecture has 13,640,064 parameters, a 256-wide hidden state,
16 transformer layers, 8 attention heads, a 1,024-wide feed-forward layer, a
128-token context, and tied input/output embeddings. It uses a fixed
multi-scale relational geometry instead of learned positional embeddings,
which keeps the architecture compact while still giving the model useful
short-range and long-range token relationships.

The important distinction is not only parameter count. NPCMicro13M separates
three jobs that are often mixed together in larger chat systems:

1. **The model learns dialogue behavior and language patterns.**
2. **The game supplies current facts through state.**
3. **The runtime enforces deployment-specific grounding and output handling.**

That separation makes it easier to reuse one model across hundreds of NPCs,
change a character without retraining the whole world, and keep server memory
and inference work bounded.

## Repository layout

```text
model/                         Fine-tuning-ready checkpoint
tokenizer/                     Matching tokenizer
training/                      Trainer, manifest tool, and example data
deployment/                    CLI, API, GUI, and grounded runtime
lineage/repaircorpus/          Exact v9 repair/preservation corpus
lineage/source/                Training and evaluation source lineage
SHA256SUMS.txt                 Frozen-artifact integrity hashes
```

The checkpoint is stored with Git LFS. Run the integrity check after cloning:

```text
python deployment/uomind_infer.py --bundle . --verify-only
```

## Limitations

NPCMicro13M is intentionally specialized for short responses. It is not a
replacement for a large reasoning model, a database, a quest engine, or a
general knowledge system. Give the game facts explicitly when accuracy
matters, and fine-tune with examples that demonstrate how the NPC should
respond when a question is outside its knowledge.

## License

This repository is distributed under the Apache License 2.0. Third-party
dependencies, source materials, and downstream datasets remain subject to
their own licenses.
