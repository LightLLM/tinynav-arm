import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from ueb.menagerie import MENAGERIE_BODIES, MenagerieEnv, ReferenceTeacher, menagerie_root
from ueb.model import BrainConfig, UnifiedBrain

try:
    menagerie_root()
except FileNotFoundError:
    pytest.skip("MuJoCo Menagerie models are not available", allow_module_level=True)

JOINTS = {"g1": 29, "ur5e": 6, "x2": 4, "go2": 12}


@pytest.mark.parametrize("body", list(MENAGERIE_BODIES))
def test_teacher_completes_the_task(body):
    env = MenagerieEnv(body)
    teacher = ReferenceTeacher(env)
    obs = env.reset(seed=3)
    assert obs.joints.shape == (JOINTS[body], 2)
    for _ in range(env.info.episode_steps):
        obs, failed = env.step(teacher.act(obs.joints))
        assert not failed
    assert env.success()


def test_g1_falls_without_control_and_reports_failure():
    env = MenagerieEnv("g1")
    obs = env.reset(seed=0)
    failed = False
    for _ in range(env.info.episode_steps):
        obs, failed = env.step(np.zeros(29, dtype=np.float32))
        if failed:
            break
    assert failed and not env.success()


def test_one_model_accepts_all_four_robot_classes():
    model = UnifiedBrain(BrainConfig(width=32, layers=1, heads=2))
    for embodiment_id, body in enumerate(MENAGERIE_BODIES):
        env = MenagerieEnv(body)
        obs = env.reset(seed=0)
        chunk = model.sample(torch.from_numpy(env.spec.joint_features), torch.from_numpy(obs.joints)[None],
                             torch.from_numpy(obs.root)[None], embodiment_id, env.spec.control_hz, steps=2)
        assert chunk.shape == (1, JOINTS[body], model.config.horizon)
        assert torch.isfinite(chunk).all()
