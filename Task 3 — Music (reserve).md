# Task 3 (reserve) — Music: implementation spec

Text-to-music and music-to-music in one model family. Built during preparation, used on stage only if the schedule allows. It is the cheapest task to build and the hardest to present.

## Models

Stable Audio 3.0, from the `stabilityai/stable-audio-3` collection on Hugging Face. Three of the four variants ship as open weights; Large does not and is excluded.

| Variant | Params | Max length | Open weights | Role |
| --- | --- | --- | --- | --- |
| 3.0 Medium | 1.4B | 6 min 20 s | Yes | DGX and workstation |
| 3.0 Small | 459M | 2 min | Yes | CPU |
| 3.0 Small SFX | 459M | short effects | Yes | Not used unless a sound-effects angle is wanted |
| 3.0 Large | 2.7B | 6 min 20 s | No, API and enterprise only | Excluded |

Confirm the exact repository ids inside the collection before downloading. Stability states the family was trained entirely on licensed data, which is the strongest licensing position of any model in this demo.

None of these models produce vocals or lyrics. Instrumental only — say so before the audience hears the first clip, or someone will ask.

## ACE-Step: now eligible, and arguably better

The origin restriction has been lifted, which puts ACE-Step in scope. On licence and on family structure it beats Stable Audio 3.0.

|  | Stable Audio 3.0 | ACE-Step |
| --- | --- | --- |
| Publisher | Stability AI | ACE Studio and StepFun |
| Licence | Stability AI Community, free commercial use under $1M revenue | Apache 2.0, no revenue threshold |
| Open tiers | Small 459M, Small SFX 459M, Medium 1.4B | v1 3.5B; the 1.5 family adds base, sft, turbo and turbo-rl variants with 0.6B / 1.7B / 4B language models |
| Vocals | No | Yes, full songs with lyrics |
| Training data | Stated to be entirely licensed | Not stated in the same terms |
| Published speed | 0.44 s (Small) and 1.31 s (Medium) on an H200 | \~20 s for 4 minutes of music on an A100; per-device RTF table on the model card |
| CPU tier | Small documented as running on CPU | Not documented |

Three notes before switching:

1. **Take ACE-Step 1.5, not v1-3.5B.** The 1.5 family is current and has the better variant ladder. Its text encoders are Qwen-based, from Alibaba, which no longer matters under the revised rules but is worth knowing.
2. **Vocals change the risk profile.** A model singing generated lyrics in front of press invites questions about training data that the ACE-Step card does not answer in the way Stability answers them. Stable Audio 3.0 avoids this entirely by producing no vocals.
3. **The published RTF figures are self-reported** on the model card and should be measured on our own hardware before being shown on stage.

ACE-Step's per-device RTF table is, in effect, a smaller version of the comparison this whole session is built on — which makes it a natural fit, and also means the audience could check our numbers against the publisher's.

## The protocol problem, and the fix

Stability reports inference of roughly 0.44 s for Small and 1.31 s for Medium on an H200. At those speeds a 30 to 60 second budget is meaningless on any GPU: the DGX and the workstation both finish instantly and tie. Iso-time as used in the other tasks does not work here.

The fix: run this task as a **throughput** demonstration. Fixed budget, count complete tracks produced.

| Machine | Model | Track length | Measure |
| --- | --- | --- | --- |
| DGX H200 | 3.0 Medium | 60 s | Tracks completed in the budget |
| Workstation 16 GB | 3.0 Medium | 60 s | Tracks completed in the budget |
| CPU 16 GB | 3.0 Small | 60 s | Tracks completed, likely one or none |

That is a real DGX advantage, a single number, and it survives the fact that one track is fast everywhere.

## Data

Two text prompts and one seed clip, frozen and deployed to all machines.

```json
[
  {"id": "mus-01", "mode": "text", "seed": 2001, "length_s": 60, "prompt": "Slow oud melody over soft percussion, warm reverb"},
  {"id": "mus-02", "mode": "text", "seed": 2002, "length_s": 60, "prompt": "Upbeat electronic track with a bright synth lead, 120 BPM"},
  {"id": "mus-03", "mode": "continuation", "seed": 2003, "length_s": 60, "audio": "inputs/seed-piano-10s.wav", "prompt": "Continue in an orchestral arrangement"}
]
```

The seed clip for `mus-03` must be our own recording or openly licensed. Not a commercial recording — at an event with press present, using an unlicensed clip in a demo about licensed training data would be an unforced error.

- [ ] Source or record the 10 s piano seed clip, and record its licence here.

## Worker contract

`POST /music/generate`

```json
{
  "run_id": "uuid",
  "input_id": "mus-01",
  "mode": "text",
  "prompt": "...",
  "audio_path": null,
  "seed": 2001,
  "length_s": 60,
  "budget_seconds": 60,
  "repeat_until_budget": true
}
```

Response:

```json
{
  "run_id": "uuid",
  "machine": "dgx",
  "model": "stable-audio-3.0-medium",
  "outputs": ["/results/uuid-0.wav", "/results/uuid-1.wav"],
  "completed": 2,
  "first_output_seconds": 1.4,
  "peak_memory_mb": 7200,
  "status": "ok"
}
```

`repeat_until_budget` is what makes the throughput measure work: the worker keeps generating with incrementing seeds until the budget expires, and reports how many finished. A partially generated track is discarded, not returned.

## How to run on each device

**DGX H200.** Medium, generating in parallel across GPUs if more than one is available. This is where the throughput number comes from. Audio output is 44.1 kHz stereo; write WAV, not a compressed format, so encoding time does not enter the measurement.

**Workstation, 16 GB VRAM.** Medium fits comfortably. Expect it to be fast but to produce far fewer tracks than the DGX in the same minute.

**CPU box, 16 GB RAM.** Small is documented as running on CPU, including on consumer laptops, so unlike the 3D task this cell should genuinely run live. Measure it: a CPU cell that works is a useful contrast with the 3D task, where it does not.

## Presentation constraints

Audio is the hardest of the three tasks to present, and these are the reasons:

| Constraint | Effect | Handling |
| --- | --- | --- |
| Audio is serial | Three clips cannot be compared at a glance | Play one clip per machine, not three; show the counts visually |
| Venue PA quality | Differences may be inaudible through house speakers | Sound check with actual generated audio, not music from a phone |
| One track is fast everywhere | Iso-time collapses | Throughput protocol above |
| No vocals | Audience may expect songs | State it upfront |
| Waveform or count display | Needed so the eye has something to follow | Show a live counter of completed tracks per machine |

## Acceptance checks before the event

- [ ] Weights for Medium and Small cached and loadable offline.
- [ ] Throughput mode works: the worker keeps generating until the budget and reports an accurate count.
- [ ] CPU box completes at least one Small track within the budget.
- [ ] Continuation mode works from the seed clip on all three machines.
- [ ] Sound check in the venue with generated audio through the house PA.
- [ ] Replay cells populated for all three inputs on all three machines.we
