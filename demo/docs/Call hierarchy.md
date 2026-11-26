# Call hierarchy

What calls what, from a button on the operator page to the model on each machine. The diagrams use the
names in the code: files are under `demo/arcdemo/` unless they start with `static/` (the pages, in
`arcdemo/conductor/static/`). They render as diagrams on GitHub and in VS Code's Markdown preview.

1. [Processes](#1-processes)
2. [Shared: one round, from Start to the summary](#2-shared-one-round-from-start-to-the-summary)
3. [Shared: Load this task's model (prepare)](#3-shared-load-this-tasks-model-prepare)
4. [Shared: when a machine fails](#4-shared-when-a-machine-fails)
5. [Task 1 — text to image](#5-task-1--text-to-image)
6. [Task 2 — image to 3D](#6-task-2--image-to-3d)
7. [Task 3 — music (reserve)](#7-task-3--music-reserve)
8. [Task 4 — language model race](#8-task-4--language-model-race)
9. [Task 8 — scene construction](#9-task-8--scene-construction)

Where the task-specific code sits:
- **Conductor:** `conductor/tasks/<task>.py` (a `TaskHandler`).
- **Worker:** `worker/tasks.py` (per-task tables) and `worker/backends/<library>.py`.
- **Pages:** `STAGE_VIEW[task]`, `OPERATOR_VIEW[task]` and `TASK_TEXT[task]`.

## 1. Processes

```mermaid
flowchart LR
    subgraph Browser
        OP["operator page<br/>static/operator.js"]
        ST["stage page<br/>static/stage.js + layout3d.js"]
    end
    subgraph Conductor["conductor (python -m arcdemo.conductor)"]
        APP["conductor/app.py<br/>create_app: shared routes<br/>+ tasks/*.routes"]
        CORE["conductor/core.py<br/>Conductor"]
        TH["conductor/tasks/*.py<br/>TaskHandler per task"]
        RS["conductor/replay.py<br/>ReplayStore"]
    end
    subgraph Machine["each machine: one worker per task (python -m arcdemo.worker)"]
        WAPP["worker/app.py<br/>create_app, JobRunner, ModelManager"]
        WT["worker/tasks.py<br/>per-task rules"]
        BE["worker/backends/*.py<br/>one Backend per library"]
        LS["llama-server<br/>(Tasks 4 and 8)"]
    end
    OP -- "POST /api/round, /api/prepare ..." --> APP
    ST -- "GET /api/state (poll)" --> APP
    APP --> CORE
    CORE --> TH
    CORE --> RS
    CORE -- "POST /jobs, GET /jobs/{id}, GET /files/{name}" --> WAPP
    WAPP --> WT
    WAPP --> BE
    BE -- "HTTP /v1/chat/completions" --> LS
```

## 2. Shared: one round, from Start to the summary

Every task follows this path. The boxes marked *handler* are where `conductor/tasks/<task>.py` changes it.

```mermaid
sequenceDiagram
    autonumber
    participant OP as operator.js
    participant APP as conductor/app.py
    participant C as Conductor (core.py)
    participant H as TaskHandler (tasks/TASK.py)
    participant W as worker/app.py
    participant JR as JobRunner._run (thread)
    participant B as Backend
    participant ST as stage.js

    OP->>APP: start() POST /api/round
    APP->>C: start_round(segment, input)
    C->>H: prepare_input(item) and caption()
    C->>C: _new_cell() per machine, then a task per cell
    par every machine at once
        C->>C: _run_cell(mid)
        C->>H: _job_request() calls job_fields()
        C->>W: POST /jobs (JobRequest)
        W->>JR: JobRunner.submit() with tasks.requested()
        JR->>JR: tasks.prepare_inputs() (before the clock)
        JR->>B: ModelManager.ensure() then Backend.load() if not loaded
        JR->>JR: tasks.settings_echo() + Backend.describe()
        JR->>B: Clock.start(), generate(req, JobContext) inside memory_meter()
        B-->>JR: ctx.record_output() per output (timestamped)
        loop every poll_every_seconds until a terminal status
            C->>W: GET /jobs/{run_id}
            W-->>C: JobRunner.snapshot()
            C->>W: _apply_result() GET /files/{name} (new outputs)
        end
        C->>H: budget_note(), unfinished(), after_result()
        C->>C: _final_seconds(), _log() to runs.jsonl
    end
    C->>C: _finish_round() then _summarise() with handler.summary()
    loop every stage.poll_ms
        ST->>APP: Demo.poll() GET /api/state
        APP-->>ST: Conductor.state()
        ST->>ST: buildRound() (STAGE_VIEW.input), updateCellStatic(), tick(), headline()
    end
```

## 3. Shared: Load this task's model (prepare)

```mermaid
sequenceDiagram
    autonumber
    participant OP as operator.js
    participant C as Conductor (core.py)
    participant W as worker/app.py (each URL of the machine)
    participant MM as ModelManager

    OP->>C: POST /api/prepare, prepare(segment)
    loop every machine
        C->>W: _free_others() POST /unload {keep_task}
        W->>MM: unload(keep_task), free_memory()
        C->>W: POST /load {model} (409 while a job ends: retried later)
        W->>MM: ensure(model) in a thread
    end
    loop _watch_prepare() every prepare_poll_seconds
        C->>W: GET /health (loaded, loading, busy)
        C->>W: GET /env (last_error) when nothing is loading
    end
    C-->>OP: prepare_state per machine: ready, missing, skipped or error
```

## 4. Shared: when a machine fails

```mermaid
flowchart TD
    R["_run_cell(mid)"] --> M{"replay mode<br/>or forced?"}
    M -- yes --> PR["_play_replay()"]
    M -- no --> D{"model on disk?<br/>(health 'available')"}
    D -- no --> FB["_fallback(reason)"]
    D -- yes --> J["POST /jobs"]
    J -- "unreachable / rejected" --> FB
    J --> P["poll GET /jobs/{id}"]
    P -- "lost contact > unreachable_after<br/>404 (worker restarted)<br/>hard_budget task past budget + grace" --> FB
    P -- "status error" --> FB
    P -- "slow machine past budget<br/>(show_full_recording)" --> SF["_show_full_recording()<br/>badge RECORDED"]
    P -- "ok, budget_exceeded,<br/>out_of_memory" --> LIVE["shown LIVE: they are results"]
    FB --> PR
    PR --> RS["ReplayStore.get()<br/>replay/&lt;task&gt;/&lt;protocol&gt;/&lt;machine&gt;/&lt;input&gt;/record.json"]
    RS -- found --> RP["badge REPLAY, outputs revealed<br/>at their rehearsal times"]
    RS -- none --> UN["status unavailable,<br/>note says why"]
```

## 5. Task 1 — text to image

`conductor/tasks/t2i.py` · `worker/backends/diffusers_t2i.py` (mock: `backends/mock.py`)

```mermaid
sequenceDiagram
    autonumber
    participant OP as operator.js
    participant T as TextToImage (tasks/t2i.py)
    participant MOD as moderation.py
    participant C as Conductor
    participant W as worker JobRunner
    participant D as DiffusersT2I
    participant ST as stage.js

    OP->>OP: fillExamples() with the spare prompts (cfg.ui.spare_prompts)
    OP->>T: POST /api/moderate (route registered by T.routes)
    T->>MOD: check_prompt(prompt), terms from config/blocklist.txt
    OP->>C: POST /api/round (input "live" + prompt)
    C->>T: prepare_input(): check_prompt again, refuse unless override
    C->>T: job_fields(): prompt
    C->>W: POST /jobs (width, height, steps, batch, seed)
    W->>D: load(): one pipeline per device, 2-step warm-up
    W->>D: generate()
    par one thread per GPU (work(k)), images round-robin
        D->>D: pipe(prompt, generator=CPU seed+i, callback_on_step_end=on_step)
        D->>D: on_step: ctx.progress(step), ctx.should_stop() aborts the image
        D->>W: ctx.save_image(i) then record_output()
    end
    C->>T: unfinished(): a partial batch shows the full recording on a slow machine
    ST->>ST: STAGE_VIEW.t2i: image grid, "3 of 4 in 60 s", headline "N images"
```

## 6. Task 2 — image to 3D

`conductor/tasks/i23d.py` · `imageprep.py` · `scoring.py` · `worker/backends/sf3d.py`, `trellis.py`

```mermaid
sequenceDiagram
    autonumber
    participant OP as operator.js
    participant T as ImageTo3D (tasks/i23d.py)
    participant IP as imageprep.py
    participant C as Conductor
    participant W as worker JobRunner
    participant S as StableFast3D / Trellis2 / Trellis1
    participant SC as scoring.py
    participant ST as stage.js

    OP->>T: loadUploadSpec() GET /api/i23d/spec (data/i23d/upload_rules.json)
    opt the operator's own picture
        OP->>T: onUpload() POST /api/i23d/upload (data URL)
        T->>IP: add_upload() calls prepare(): checks, u2net_mask() cut-out, resize_foreground()
        T->>C: cfg.add_input("i23d", item), uploads.json
    end
    OP->>C: POST /api/round
    C->>T: caption(): no score for an upload
    C->>T: job_fields(): image_path, or image_b64 for an upload
    C->>W: POST /jobs
    W->>W: tasks.prepare_inputs(): save_upload() before the clock
    W->>S: generate()
    alt SF3D
        S->>S: resize_foreground(), model.run_image(), mesh.export(.glb)
    else TRELLIS.2 / TRELLIS v1
        S->>S: pipeline.run(seed), then o_voxel / postprocessing to_glb()
    end
    S->>W: generation_seconds, export_seconds, record_output(.glb)
    C->>C: _apply_result() fetches the .glb
    C->>T: budget_note() "Finished after the bell" (soft budget)
    C->>T: after_result()
    T->>SC: score_files(mesh, GSO scan) in a thread (Chamfer, F-score)
    ST->>ST: STAGE_VIEW.i23d: input picture, model-viewer, "Error vs scan"
```

## 7. Task 3 — music (reserve)

`conductor/tasks/music.py` · `worker/backends/stable_audio3.py` (mock: `backends/music_stub.py`). There is
no segment in `demo.yaml` yet, so the operator cannot start it; the path is ready.

```mermaid
sequenceDiagram
    autonumber
    participant C as Conductor
    participant T as TextToMusic (tasks/music.py)
    participant W as worker JobRunner
    participant A as StableAudio3
    participant ST as stage.js

    C->>T: job_fields(): prompt, mode, audio_path, length_s
    C->>W: POST /jobs (tasks.requested = 0: throughput)
    W->>A: load(): StableAudioModel.from_pretrained(sa3_name, precision)
    W->>A: generate()
    loop until the budget (repeat_until_budget)
        A->>A: _generate_track(): model.generate(prompt, duration, steps, seed+i)
        A->>A: write_wav(), then record_output(), or discard a track that ended after the bell
    end
    ST->>ST: STAGE_VIEW.music (shared): audio player, track count
```

## 8. Task 4 — language model race

`conductor/tasks/llm.py` · `prompts.py` · `worker/backends/llm.py` (mock: `LlmMock`)

```mermaid
sequenceDiagram
    autonumber
    participant OP as operator.js
    participant T as LanguageModel (tasks/llm.py)
    participant C as Conductor
    participant W as worker JobRunner
    participant L as LlamaServer
    participant LS as llama-server
    participant ST as stage.js

    OP->>T: showPrompt() GET /api/prompt, prompt_text() with prompts.llm_system()
    OP->>C: POST /api/round
    C->>T: job_fields(): prompt
    C->>W: POST /jobs (streams, max_tokens, temperature, repeat_until_budget)
    W->>L: load(): snapshot_download(GGUF), Popen llama-server, wait GET /health, warm-up
    W->>L: LlmBackend.generate()
    L->>L: run_streams(): one thread per stream, guarded(i), run(i)
    loop each stream, until the reply ends or the bell
        L->>L: _stream() then _chat(messages)
        L->>LS: _open() POST /v1/chat/completions (stream)
        LS-->>L: data: lines, one token per chunk, then a usage count
        L->>L: count(): tokens on the job clock, refused after the bell
    end
    opt Stop or the bell
        L->>L: _abort_streams(), _hang_up() shuts the socket at once
    end
    L->>W: record_output() per finished reply, tokens_per_second
    ST->>ST: drawText() types the reply, token counter, "N tok/s"
```

## 9. Task 8 — scene construction

`conductor/tasks/layout.py` · `layout_prompt.py` · `worker/backends/layout.py` (mock: `LayoutMock`) ·
`static/layout3d.js`

```mermaid
sequenceDiagram
    autonumber
    participant OP as operator.js
    participant T as SceneLayout (tasks/layout.py)
    participant LP as layout_prompt.py
    participant C as Conductor
    participant W as worker JobRunner
    participant LB as LayoutServer / _Room
    participant LS as llama-server
    participant ST as stage.js + layout3d.js

    opt the operator's own room
        OP->>T: loadRoomRules() GET /api/layout/rules
        OP->>T: previewRoom() POST /api/layout/room (save false): check_room(), layout_view()
        OP->>T: addRoom() POST /api/layout/room: add_room(), cfg.add_input("layout", item), rooms.json
    end
    OP->>T: loadLayoutView() GET /api/layout_view, then layout_view(): wall names, request in words
    OP->>T: showPrompt() GET /api/prompt, then layout_prompt_text()
    T->>LP: messages() or layoutgpt_messages(), grammar() or css_grammar()
    ST->>T: GET /api/layout_catalog (furniture models at /layout-assets)
    OP->>C: POST /api/round
    C->>T: prepare_input(): layout_view() into the round's input
    C->>T: job_fields(): room_type, floor_plan, door, max_objects
    C->>W: POST /jobs (+ repairs, review_rounds, layout_prompt)
    W->>LB: load(): LlamaServer.load(), then one warm-up request per slot (shared prompt prefix)
    W->>LB: LayoutBackend.generate(): room_spec(), run_streams()
    loop each stream, each room
        LB->>LB: _Room.run() then build()
        LB->>LP: continue_messages(), grammar()
        LB->>LS: stream() then _reply() then _chat(grammar=...)
        LB->>LP: parse_line() (or parse_css), check(): outside, overlap, door
        LB->>LB: publish(): result.objects with times
        opt self-correction (Furnish and fix)
            LB->>LP: repair(): repair_messages(), repair_grammar(), problem_cost()
            LB->>LP: review(): review_messages(), edit_grammar(), parse_edit(), room_cost()
        end
        LB->>W: room JSON, then record_output()
    end
    C->>T: summary() then layout_summary(): objects, flags, fixes
    ST->>ST: Layout.attach() / draw() (3D, or the layout.js canvas), drawConsole()
```
