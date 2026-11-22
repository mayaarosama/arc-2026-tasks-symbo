"""Layout parsing (prompt v3, wall-anchored), grammar and the room checks (no worker, no model)."""
import re

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
