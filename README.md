# NPCMicro13M

NPCMicro13M is a self-contained, CPU-friendly conversational language model
for short Ultima Online-style NPC interactions. It contains the
trainer-compatible v9 base checkpoint (13,640,064 parameters), matching
tokenizer, the exact preservation-first trainer used for the v9 continuation,
the corpus shards used in that continuation, source-lineage files, and
deployment tools.

The checkpoint is inference/training ready but does not include optimizer
state, epochs, metrics, or other training-session bloat. It retains the full
model state, including both tied embedding keys, because the fine-tuning
trainer expects the complete state dictionary.

## 1. Install

Use Python 3.10 or newer:

```text
python -m pip install -r requirements.txt
```

For NVIDIA CUDA, install the appropriate PyTorch build first using the command
recommended by PyTorch, then install `tokenizers`.

## 2. Verify and run the base model

From this release folder:

```text
python deployment/uomind_infer.py --bundle . --verify-only
python deployment/uomind_infer.py --bundle . --state "Your name is Mira. You are a baker in Britain." --player "Who are you?"
```

The default response uses the grounded response layer. To inspect the raw
language-model output:

```text
python deployment/uomind_infer.py --bundle . --raw-model --state "Your name is Mira. You are a baker in Britain." --player "Who are you?"
```

For the GUI, double-click `deployment/uomind_gui.pyw` or
`deployment/START_UO_MIND_GUI.bat`. The GUI keeps a visible transcript, but
each model turn is single-pass: only the fixed NPC state and current player
message are sent to the model.

## 3. Fine-tuning data format

Each training row is one JSON object on one line:

```json
{"messages":[{"role":"system","content":"Your name is Mira. You are a baker in Britain."},{"role":"user","content":"Who are you?"},{"role":"assistant","content":"I am Mira, a baker in Britain."}],"meta":{"category":"identity","family":"identity_001","pair_id":null,"preservation":true}}
```

Required fields:

- `messages`: exactly `system`, `user`, `assistant`, in that order.
- `meta.category`: a descriptive task category.
- `meta.family`: a grouping name used to keep related examples together.
- `meta.preservation`: `true` for behavior that should remain intact; `false` for new behavior.

Use several differently worded examples for each behavior. For a task-specific
fine-tune, include preservation examples from the base model so the new task
does not erase identity, formatting, safety, or conversational behavior.

An editable six-row example is in
`training/corpus/example_custom_task.jsonl`. Copy it to a new filename, edit
the rows, and create its manifest:

```text
python training/make_manifest.py training/corpus/my_task.jsonl
```

## 4. Reproduce the v9 continuation

The exact v9 repair/preservation corpus is in
`lineage/repaircorpus/`. To replay that training path from the v9 base:

```text
python training/uomind_repair_finetune_v3.py ^
  --bundle . ^
  --checkpoint model/v9_base_finetune.pt ^
  --allow-source-mismatch ^
  --corpus-dir lineage/repaircorpus ^
  --corpus-glob "uomind_preservation_repair_corpus_part*.jsonl" ^
  --require-manifests ^
  --output-dir outputs/v9_replay ^
  --train ^
  --device cpu
```

PowerShell users can put the command on one line or replace `^` with a
backtick. The trainer defaults to a low-cost selective update of the final
four transformer blocks. Start with one epoch on CPU. The original v9 run used
five shards containing 3,241 rows: 2,000 repair rows and 1,241 preservation
rows.

## 5. Fine-tune for a new task

Put your manifest-backed JSONL shard in `training/corpus/`, then run:

```text
python training/uomind_repair_finetune_v3.py ^
  --bundle . ^
  --checkpoint model/v9_base_finetune.pt ^
  --allow-source-mismatch ^
  --corpus-dir training/corpus ^
  --corpus-glob "my_task.jsonl" ^
  --require-manifests ^
  --output-dir outputs/my_task ^
  --train ^
  --device cpu ^
  --epochs 1 ^
  --train-last-n-layers 4
```

For a GPU, change `--device cpu` to `--device cuda` and choose a suitable
PyTorch CUDA installation. To make a stronger update, use
`--train-last-n-layers 0` to train all transformer blocks. Only enable
`--train-embedding` when the task genuinely needs new vocabulary behavior;
that option is more likely to change existing responses.

The trainer writes checkpoints under `outputs/my_task/checkpoints/`:

- `best_preserved.pt`: the preferred candidate when it passes preservation gates.
- `final_unpromoted.pt`: the final diagnostic candidate, even if it regressed.

Do not overwrite `model/v9_base_finetune.pt`. Keep every experiment in its own
output folder and manually inspect the raw responses before deployment.

## 6. Use a fine-tuned checkpoint

One-shot CLI inference:

```text
python deployment/uomind_infer.py ^
  --bundle . ^
  --checkpoint outputs/my_task/checkpoints/best_preserved.pt ^
  --state "Your name is Nessa. You are a merchant in Britain." ^
  --player "What do you sell?"
```

Programmatic use:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("deployment").resolve()))
from uomind_api import UOMindRuntime

runtime = UOMindRuntime(
    bundle=".",
    checkpoint="outputs/my_task/checkpoints/best_preserved.pt",
    device="cpu",
)
answer = runtime.respond(
    "Your name is Nessa. You are a merchant in Britain.",
    "What do you sell?",
)
print(answer["text"])
```

`respond()` does not accept conversation history. This is intentional: the
model was trained for single-pass inference using `STATE` plus the current
`PLAYER` message.

## 7. Release layout

- `model/v9_base_finetune.pt`: trainer-compatible v9 base model.
- `tokenizer/tokenizer.json`: matching 4,096-token tokenizer.
- `training/uomind_repair_finetune_v3.py`: exact v9 continuation trainer.
- `training/make_manifest.py`: helper for custom JSONL manifests.
- `training/corpus/`: editable custom-task example.
- `lineage/repaircorpus/`: the five corpus shards used for v9.
- `lineage/source/`: original Colab training and evaluation source files.
- `lineage/v9_training_summary.json`: v9 run provenance and corpus audit.
- `deployment/`: CLI, API, GUI, grounded layer, and launcher.
- `SHA256SUMS.txt`: integrity hashes for the release.
- `LICENSE`: Apache License 2.0 for this repository.

The release is intentionally self-contained and does not include the original
172 MB frozen production checkpoint or training caches; the v9 base checkpoint
is the model intended for continued fine-tuning.

## 8. License and responsible use

The repository is distributed under the Apache License 2.0. Third-party
dependencies, source materials, and any downstream datasets remain subject to
their own licenses. Review the included `LICENSE` and the provenance files
before redistributing or deploying a fine-tuned model.
