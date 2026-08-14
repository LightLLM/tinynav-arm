from tinynav.controller import SafetyController
from tinynav.types import Action, Detection


controller = SafetyController()


def test_clear_path_goes() -> None:
    assert controller.decide([]).action is Action.GO


def test_close_obstacle_stops() -> None:
    detection = Detection("crate", 0.99, (200, 100, 400, 300), 0.6)
    assert controller.decide([detection]).action is Action.STOP


def test_caution_zone_slows() -> None:
    detection = Detection("forklift", 0.96, (200, 100, 400, 300), 2.0)
    assert controller.decide([detection]).action is Action.SLOW


def test_low_confidence_is_ignored() -> None:
    detection = Detection("ghost", 0.10, (200, 100, 400, 300), 0.2)
    assert controller.decide([detection]).action is Action.GO
