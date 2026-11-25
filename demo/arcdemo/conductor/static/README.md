# arcdemo/conductor/static/: the operator and stage pages

Plain HTML, CSS and JavaScript, served by the conductor at `/static`. There is no build step. Both pages
poll `GET /api/state`. Their own settings come from `GET /api/config` (`ui`: poll intervals, view names,
terminal statuses, the spare-prompts file), which `config/defaults.yaml` `stage` sets.

| File | What |
| --- | --- |
| `operator.html` / `operator.js` | The operator console. Shared code first: choosing a part and an input, *What will run*, start / stop, the live state table. Then one section per task filling `OPERATOR_VIEW[task]`: Task 1 batch count, Task 2 picture preview and upload box, Task 4 / 8 prompt view, Task 8 floor-plan preview |
| `stage.html` / `stage.js` / `stage.css` | The projector page. Shared rendering first: round header, cells, clocks, headline, summary chart. Then one section per task filling `STAGE_VIEW[task]`. Task 1 has image counts; Task 2 the input picture and score; Task 4 streamed text and token counters (`drawText`); Task 8 the 3D room, the model's lines (`drawConsole`) and rooms-per-minute |
| `common.js` | Shared by both pages: state polling (`Demo`), clock maths, formatting, `el()`; then `TASK_TEXT[task]` (settings and expected output in words) in show order, the Task 4 token formatting, and the Task 8 plan drawing (`planSvg`, `layoutLine`) |
| `layout.js` | Task 8: the 2D canvas view of a room, used when WebGL or the 3D module is not available |
| `layout3d.js` | Task 8: the 3D room (three.js ES module). It loads the furniture models from `GET /api/layout_catalog` and takes over from `layout.js` once loaded |
| `vendor/` | Third-party, vendored so the show needs no network: `model-viewer.min.js` (Task 2 meshes), `three/` (Task 8, MIT, with its licence) |
| `assets/furniture/` | The Kenney Furniture Kit (CC0, licence included): the 25 `.glb` models `data/layout/furniture.json` names |

`stage.html` must load the import map before any module script, or `three` cannot be resolved; see the
comment in the file. `tests/test_data_and_static.py` checks this and that every file a page loads exists.
