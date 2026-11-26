"""Build the rehearsal prompt set from PartiPrompts (Yu et al., 2022), deterministically.

The stage uses three fixed prompts. Rehearsal needs more: to measure run-to-run timing spread,
and to estimate how often an unrehearsed (audience) prompt produces something unusable.
We sample by PartiPrompts "Challenge" — the categories where small models visibly fail —
and exclude the People category so no rehearsal output depicts a person.

    python -m scripts.t2i.build_rehearsal_prompts path/to/PartiPrompts.tsv
Source TSV: https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts.tsv
"""
import csv
import json
import random
import sys
from pathlib import Path

from scripts.common.settings import SCRIPTS

B = SCRIPTS.build_rehearsal_prompts     # config/scripts.yaml
CHALLENGES = B.challenges
PER_CHALLENGE = B.per_challenge
SEED = B.seed


def main(tsv: str, out: str = B.out) -> None:
    rows = list(csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t"))
    rng = random.Random(SEED)
    picked = []
    for ch in CHALLENGES:
        pool = [r for r in rows if r["Challenge"] == ch and r["Category"] != "People"
                and B.min_words <= len(r["Prompt"].split()) <= B.max_words]
        for r in rng.sample(pool, PER_CHALLENGE):
            picked.append(r)
    data = {
        "_source": f"PartiPrompts (google-research/parti), sampled by scripts/t2i/build_rehearsal_prompts.py, seed {SEED}",
        "prompts": [{"id": f"parti-{i+1:02d}", "seed": B.first_prompt_seed + i, "challenge": r["Challenge"],
                     "category": r["Category"], "prompt": r["Prompt"]} for i, r in enumerate(picked)],
    }
    Path(out).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(picked)} prompts to {out}")



if __name__ == "__main__":
    main(*sys.argv[1:])
