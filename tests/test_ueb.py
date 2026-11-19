import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")

from tinynav.types import Action
from ueb.embodiment import JOINT_FEATURES, ROOT_FEATURES
from ueb.env import BodyEnv
from ueb.model import BrainConfig, UnifiedBrain
from ueb.pipeline import BrainPolicy, collect, rollout, train
from ueb.safety import PRESETS, ReflexLayer


@pytest.fixture(scope="module")
def humanoid():
    env = BodyEnv("humanoid")
    yield env
    env.close()


def test_humanoid_is_described_per_joint(humanoid):
    obs = humanoid.reset(seed=1)
    assert humanoid.spec.num_joints == 17
    assert humanoid.spec.joint_features.shape == (17, JOINT_FEATURES)
    assert obs.joints.shape == (17, 2)
    assert obs.root.shape == (ROOT_FEATURES,)
    assert obs.upright > 0.9 and obs.height > 1.0


def test_same_weights_drive_bodies_with_different_joint_counts(humanoid):
    model = UnifiedBrain(BrainConfig(width=32, layers=1, heads=2))
    before = sum(p.numel() for p in model.parameters())
    for embodiment_id, body in enumerate(["humanoid", "walker", "ant"]):
        env = humanoid if body == "humanoid" else BodyEnv(body)
        obs = env.reset(seed=0)
        features = torch.from_numpy(env.spec.joint_features)
        chunk = model.sample(features, torch.from_numpy(obs.joints)[None], torch.from_numpy(obs.root)[None],
                             embodiment_id, env.spec.control_hz, steps=2)
        assert chunk.shape == (1, env.spec.num_joints, model.config.horizon)
        assert torch.isfinite(chunk).all()
    assert sum(p.numel() for p in model.parameters()) == before


def test_training_reduces_the_action_loss():
    data = collect("humanoid", 0, episodes=3, horizon=8, max_steps=60)
    config = BrainConfig(width=32, layers=1, heads=2)
    generator = torch.Generator().manual_seed(0)
    batch = data.batch(256, generator)

    def action_loss(model):
        torch.manual_seed(0)
        with torch.no_grad():
            return model.loss(batch, data.joint_features, 0, data.env.spec.control_hz)[1]["action"]

    torch.manual_seed(0)
    untrained = action_loss(UnifiedBrain(config))
    trained = action_loss(train([data], steps=150, batch_size=64, config=config))
    assert trained < untrained * 0.7

    stats = rollout(data.env, BrainPolicy(train([data], steps=5, config=config), data.env, 0), episodes=1, max_steps=20)
    assert 1 <= stats.mean_steps <= 20


def test_reflex_passes_safe_commands_and_limits_large_ones(humanoid):
    reflex = ReflexLayer(humanoid.spec, PRESETS["humanoid"])
    mid_range = (humanoid.spec.joint_lower + humanoid.spec.joint_upper) / 2
    small = np.full(17, 0.05, dtype=np.float32)
    assert reflex.filter(small, mid_range, 1.3, 1.0).action is Action.GO

    reflex.reset()
    decision = reflex.filter(np.full(17, 5.0, dtype=np.float32), mid_range, 1.3, 1.0)
    assert decision.action is Action.SLOW
    assert np.abs(decision.command).max() <= PRESETS["humanoid"].max_delta + 1e-6


def test_reflex_never_pushes_into_a_joint_limit(humanoid):
    reflex = ReflexLayer(humanoid.spec, PRESETS["humanoid"])
    at_upper = humanoid.spec.joint_upper.copy()
    decision = reflex.filter(np.full(17, 0.1, dtype=np.float32), at_upper, 1.3, 1.0)
    assert (decision.command <= 0).all()


def test_reflex_stops_on_fall_and_stays_stopped(humanoid):
    reflex = ReflexLayer(humanoid.spec, PRESETS["humanoid"])
    pos = np.zeros(17, dtype=np.float32)
    fallen = reflex.filter(np.full(17, 0.1, dtype=np.float32), pos, 0.3, 0.1)
    assert fallen.action is Action.STOP and not fallen.command.any()
    recovered_reading = reflex.filter(np.full(17, 0.1, dtype=np.float32), pos, 1.3, 1.0)
    assert recovered_reading.action is Action.STOP


def test_reflex_stops_on_non_finite_command(humanoid):
    reflex = ReflexLayer(humanoid.spec, PRESETS["humanoid"])
    bad = np.full(17, np.nan, dtype=np.float32)
    assert reflex.filter(bad, np.zeros(17, dtype=np.float32), 1.3, 1.0).action is Action.STOP
