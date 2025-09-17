from cabaret.focuser import Focuser

BEST_POSITION = 10_000
SCALE = 100


def test_focuser_default_multiplier():
    focuser = Focuser()
    assert focuser.seeing_multiplier == 1.0


def test_focuser_in_focus():
    focuser = Focuser(best_position=BEST_POSITION)
    assert focuser.seeing_multiplier == 1.0


def test_focuser_out_of_focus():
    focuser = Focuser(
        position=BEST_POSITION + SCALE, best_position=BEST_POSITION, scale=SCALE
    )
    # offset = SCALE, multiplier = 1 + SCALE/SCALE = 2.0
    assert focuser.seeing_multiplier == 2.0


def test_focuser_max_multiplier():
    focuser = Focuser(
        position=BEST_POSITION + 5 * SCALE,
        best_position=BEST_POSITION,
        scale=SCALE,
        max_seeing_multiplier=2.0,
    )
    # offset = 5*SCALE, multiplier = 1 + 5*SCALE/SCALE = 6.0, but max is 2.0
    assert focuser.seeing_multiplier == 2.0


def test_focuser_negative_offset():
    focuser = Focuser(
        position=BEST_POSITION - 2 * SCALE, best_position=BEST_POSITION, scale=100
    )
    # offset = 2*SCALE, multiplier = 1 + 2*SCALE/SCALE = 3.0
    assert focuser.seeing_multiplier == 3.0
