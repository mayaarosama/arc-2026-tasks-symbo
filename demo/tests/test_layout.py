"""Layout parsing (prompt v3, wall-anchored), grammar and the room checks (no worker, no model)."""
import re

from arcdemo import layout_prompt as P
from arcdemo.worker.backends import layout as L

ROOM = {"room_type": "bedroom", "floor_plan": [[0, 0], [4, 0], [4, 3], [0, 3]], "door": [3.5, 0], "max_objects": 6}
ELL = {"room_type": "living_room", "floor_plan": [[0, 0], [6, 0], [6, 2.8], [3.5, 2.8], [3.5, 5], [0, 5]],
       "door": None, "max_objects": 10}


def obj(cls="desk", x=1.0, y=1.0, w=1.2, d=0.6, rot=0):
    return {"cls": cls, "x": x, "y": y, "w": w, "d": d, "h": 0.75, "rot": rot}


def test_wall_line_resolves_against_the_named_wall():
    o = L.parse_line("wardrobe wall=east at=1.70", ROOM)          # east wall x=4, wardrobe 1.2 x 0.6
    assert (o["x"], o["y"], o["rot"], o["wall"]) == (3.7, 1.7, 270, "east")
    o = L.parse_line("double_bed wall=north at=2.00", ROOM)       # north wall y=3, bed depth 2.0
    assert (o["x"], o["y"], o["rot"]) == (2.0, 2.0, 180)
    assert L.check(o, [], ROOM) == []


def test_free_line_takes_centre_and_facing_with_fixed_size():
    o = L.parse_line("chair x=1.20 y=0.90 facing=west", ROOM)
    assert (o["x"], o["y"], o["rot"], o["w"], o["d"]) == (1.2, 0.9, 270, 0.5, 0.5)


def test_reject_unknown_wall_wrong_room_and_garbage():
    assert L.parse_line("wardrobe wall=north2 at=1.00", ROOM) is None   # no such wall in this room
    assert L.parse_line("sofa wall=north at=1.00", ROOM) is None        # not a bedroom object
    assert L.parse_line("the bed goes here", ROOM) is None


def test_l_shaped_room_walls_are_named_and_numbered():
    names = L.walls(ELL["floor_plan"])
    assert list(names) == ["south", "east1", "north1", "east2", "north2", "west"]
    o = L.parse_line("cabinet wall=north1 at=4.50", ELL)               # inner wall y=2.8, floor to the south
    assert (o["y"], o["rot"]) == (2.575, 180) and L.check(o, [], ELL) == []


def test_sideways_footprint_swaps_width_and_depth():
    assert L.footprint(obj(x=1, y=1, w=1.2, d=0.6, rot=90)) == (0.7, 0.4, 1.3, 1.6)


def test_checks_flag_outside_overlap_and_door():
    assert L.check(obj(x=1.0, y=1.0), [], ROOM) == []
    assert L.check(obj(x=0.3, y=1.0), [], ROOM) == ["outside"]                  # x from -0.3
    assert L.check(obj(x=1.2, y=1.1), [obj(x=1.0, y=1.0)], ROOM) == ["overlap"]
    assert L.check(obj(x=3.3, y=0.5), [], ROOM) == ["door"]
    assert L.check(L.parse_line("wardrobe wall=east at=0.20", ROOM), [], ROOM) == ["outside", "door"]


def test_l_shaped_room_inside_corner_is_outside():
    assert L.check(obj(x=2.0, y=4.0), [], ELL) == []
    assert L.check(obj(x=4.5, y=4.0), [], ELL) == ["outside"]                   # in the missing quarter


def test_worked_examples_obey_their_own_rules():
    for spec, reply in L.EXAMPLES:
        placed = []
        for line in reply.splitlines():
            o = L.parse_line(line, spec)
            assert o is not None, line
            assert L.check(o, placed, spec) == [], line
            placed.append(o)


def test_grammar_splits_wall_and_free_pieces_and_names_this_rooms_walls():
    g = L.grammar(ELL)
    rule = lambda name: set(re.findall(r'"([a-z_0-9]+)"', g.split(f"\n{name} ::=")[1].split("\n")[0]))
    assert "root ::= line{1,10}" in g
    assert rule("wcls") == {c for c in L.ROOMS["living_room"] if c in L.WALL_PIECES}
    assert rule("fcls") == {c for c in L.ROOMS["living_room"] if c not in L.WALL_PIECES}
    assert rule("wall") == set(L.walls(ELL["floor_plan"]))


def test_mock_layout_is_clean_and_deterministic():
    a, b = L.mock_layout(ROOM, 0), L.mock_layout(ROOM, 0)
    assert a == b and 1 <= len(a) <= 6
    placed = []
    for line in a:
        o = L.parse_line(line.strip(), ROOM)
        assert o is not None and L.check(o, placed, ROOM) == []
        placed.append(o)


# ---------------------------------------------------------------- self-correction -------------
def test_measure_says_how_much_and_check_is_unchanged():
    sofa = {**obj(cls="sofa", x=1.0, y=1.0), "i": 2}
    probs = L.measure(obj(x=1.2, y=1.1), [sofa], ROOM)
    assert [p[0] for p in probs] == ["overlap"] and "3 sofa" in probs[0][2] and abs(probs[0][1] - 0.5) < 1e-9
    kinds = [p[0] for p in L.measure(L.parse_line("wardrobe wall=east at=0.20", ROOM), [], ROOM)]
    assert kinds == ["outside", "door"]
    out = L.measure(obj(x=0.3, y=1.0), [], ROOM)[0]
    assert out[0] == "outside" and abs(out[1] - 0.3) < 1e-9 and "0.30 m" in out[2]


def test_room_cost_is_zero_only_for_a_clean_room_and_removing_is_not_free():
    spec, reply = L.EXAMPLES[0]
    room = [dict(L.parse_line(line, spec), i=n) for n, line in enumerate(reply.splitlines())]
    assert L.room_cost(room, spec, len(room)) == 0
    assert L.room_cost(room[:1] + room[2:], spec, len(room)) == L.REMOVE_COST       # took the chair out
    assert L.room_cost(room[1:], spec, len(room)) == L.REMOVE_COST + L.MUST_HAVE_COST   # the desk: a study's main piece
    clash = room + [dict(L.parse_line("chair x=1.20 y=0.85 facing=north", spec), i=len(room))]
    assert L.room_cost(clash, spec, len(clash)) > 0


def test_line_of_round_trips_every_example_line():
    for spec, reply in L.EXAMPLES:
        for line in reply.splitlines():
            assert L.line_of(L.parse_line(line, spec)) == line


def test_repair_grammar_keeps_the_same_object():
    g = L.repair_grammar(ROOM, "wardrobe")
    assert g.startswith('root ::= "wardrobe wall=" wall') and '"east"' in g
    assert L.repair_grammar(ROOM, "chair").startswith('root ::= "chair x="')


def test_continuation_grammar_may_write_no_more_lines():
    assert "root ::= line{0,3}" in L.grammar(ROOM, lines=3, at_least=0)
    assert L.grammar(ROOM) == L.grammar(ROOM, lines=ROOM["max_objects"], at_least=1)


def test_edit_grammar_and_parser():
    room = [dict(L.parse_line("wardrobe wall=east at=1.70", ROOM), i=0),
            dict(L.parse_line("chair x=1.20 y=0.90 facing=west", ROOM), i=1)]
    g = L.edit_grammar(ROOM, room)
    assert 'widx ::= "1"' in g and 'fidx ::= "2"' in g and g.splitlines()[0].startswith('root ::= edit{0,6} "done')
    kind, o, new = L.parse_edit("move 1 wall=north at=2.00", room, ROOM)
    assert kind == "move" and o is room[0] and (new["wall"], new["rot"]) == ("north", 180)
    assert L.parse_edit("move 2 x=2.00 y=2.00 facing=east", room, ROOM)[2]["rot"] == 90
    assert L.parse_edit("remove 2", room, ROOM) == ("remove", room[1])
    assert L.parse_edit("done", room, ROOM) == ("done",)
    assert L.parse_edit("remove 7", room, ROOM) is None and L.parse_edit("move 2 wall=east at=1.0", room, ROOM) is None


def test_review_message_numbers_the_room_and_names_each_problem():
    room = [dict(L.parse_line("double_bed wall=north at=2.00", ROOM), i=0, issues=[]),
            dict(L.parse_line("double_bed wall=north at=2.20", ROOM), i=1, issues=["overlap"])]
    text = L.review_messages(ROOM, ["a\n", "b\n"], room, [])[-1]["content"]
    assert "1 double_bed wall=north at=2.00  ok" in text
    assert "2 double_bed wall=north at=2.20  <- problem: it overlaps 1 double_bed by" in text


def test_layout_server_keeps_the_llama_server_stream():
    """LayoutServer's reply hook must not shadow LlamaServer._stream (load()'s warm-up uses it)."""
    from arcdemo.worker.backends.llm import LlamaServer
    assert L.LayoutServer._stream is LlamaServer._stream


def test_fewer_problems_always_cost_less_than_more_small_ones():
    big = obj(x=3.3, y=0.5)                                   # by the doorway: one problem, 0.6 m deep
    assert L.check(big, [], ROOM) == ["door"]
    small = {**ROOM, "door": [0.2, 0]}
    three = obj(x=0.55, y=0.33)                               # just out, near the door, a little overlap
    assert L.check(three, [obj(x=1.5, y=0.33)], small) == ["outside", "door", "overlap"]
    assert L.problem_cost(big, [], ROOM) < L.problem_cost(three, [obj(x=1.5, y=0.33)], small)


# ---------------------------------------------------------------- LayoutGPT variant -----------
def test_css_line_round_trips_to_the_same_object():
    o = L.parse_line("wardrobe wall=east at=1.70", ROOM)
    line = L.css_line(o)
    assert line == ("wardrobe {length: 1.20m; width: 0.60m; height: 2.00m; orientation: 270 degrees; "
                    "left: 3.70m; top: 1.70m; depth: 1.00m;}")
    back = L.parse_css(line, ROOM)
    assert {k: back[k] for k in ("cls", "x", "y", "rot")} == {k: o[k] for k in ("cls", "x", "y", "rot")}
    assert back["written_size"] == [1.2, 0.6, 2.0] and back["css"] == line


def test_css_parser_snaps_facing_keeps_written_size_and_rejects_others():
    o = L.parse_css("desk {length: 2.00m; width: 1.00m; height: 0.75m; orientation: -95 degrees; "
                    "left: 1.00m; top: 1.00m; depth: 0.38m;}", ROOM)
    assert o["rot"] == 270 and (o["w"], o["d"]) == L.FURNITURE["desk"][:2] and o["written_size"] == [2.0, 1.0, 0.75]
    assert L.parse_css("sofa {length: 2.00m; width: 1.00m; height: 0.75m; orientation: 0 degrees; "
                       "left: 1.00m; top: 1.00m; depth: 0.38m;}", ROOM) is None     # not a bedroom object
    assert L.parse_css("desk wall=north at=1.00", ROOM) is None


def test_css_grammar_ends_each_line_with_an_escaped_newline():
    g = L.css_grammar(ROOM)
    assert len(g.splitlines()) == 5 and g.splitlines()[1].endswith('m;}\\n"')   # GBNF escape, not a newline


def test_example_bank_is_clean_and_has_no_stage_room():
    import json
    stage = {json.dumps(r["floor_plan"]) for r in json.load(open("data/layout/stage_rooms.json"))}
    for e in P.exemplars():
        assert json.dumps(e["floor_plan"]) not in stage
        placed = []
        for v3, css in zip(e["v3"], e["layout"]):
            o = L.parse_line(v3, e)
            assert L.check(o, placed, e) == [] and L.css_line(o) == css, (e["id"], v3)
            placed.append(o)


def test_layoutgpt_prompt_gives_room_size_only_and_k_similar_examples():
    msgs = P.layoutgpt_messages(ROOM)
    assert msgs[0]["role"] == "system" and msgs[0]["content"].startswith("You are a 3D indoor scene designer.")
    assert len(msgs) == 2 + 2 * P.LAYOUTGPT_K
    q = msgs[-1]["content"]
    assert q == "Condition:\nRoom Type: bedroom\nRoom Size: max length 4.00m, max width 3.00m\nLayout:\n"
    shots = P.similar_rooms(ROOM)
    assert all(e["room_type"] == "bedroom" for e in shots)
    assert shots[0]["floor_plan"][2] == [4.0, 3.4] or shots[0]["floor_plan"][2][0] <= 4.2   # nearest size first


def test_layoutgpt_with_self_correction_is_refused():
    from arcdemo.schemas import JobRequest
    req = JobRequest(task="layout", model="m", input_id="x", floor_plan=ROOM["floor_plan"], room_type="bedroom",
                     layout_prompt="layoutgpt", repairs=1)
    import pytest
    with pytest.raises(ValueError, match="v3.1 prompt only"):
        L.room_spec(req)
