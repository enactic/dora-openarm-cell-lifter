import importlib
import math
import sys
import types
from pathlib import Path

import pytest


sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)


class FakeValue:
    def __init__(self, value):
        self._value = value

    def as_py(self):
        return self._value


class FakeNode:
    def __init__(self, events):
        self._events = events
        self.outputs = []

    def __iter__(self):
        return iter(self._events)

    def send_output(self, name, value):
        self.outputs.append((name, value))


class FakeMotor:
    def __init__(self, lifter):
        self._lifter = lifter

    def get_position(self):
        return self._lifter.state["position"]

    def get_torque(self):
        return self._lifter.state["torque"]


class FakeArm:
    def __init__(self, lifter):
        self._lifter = lifter
        self.commands = []

    def get_motors(self):
        return [FakeMotor(self._lifter)]

    def posvel_control_all(self, params):
        self.commands.append(params)


class FakeLifter:
    def __init__(self, calibration_state, normal_state):
        self.state = {"position": 0.0, "torque": 0.0}
        self._calibration_state = calibration_state
        self._normal_state = normal_state
        self._recv_calls = 0
        self._arm = FakeArm(self)

    def get_arm(self):
        return self._arm

    def recv_all(self):
        self._recv_calls += 1
        if self._recv_calls <= 5:
            self.state = self._calibration_state
        else:
            self.state = self._normal_state


@pytest.fixture
def main_module(monkeypatch):
    fake_dora = types.SimpleNamespace(Node=None)
    fake_openarm = types.SimpleNamespace(
        PosVelParam=lambda *, q, dq: types.SimpleNamespace(q=q, dq=dq)
    )
    fake_pyarrow = types.SimpleNamespace(
        array=lambda values, type=None: list(values),
        float32=lambda: "float32",
    )

    monkeypatch.setitem(sys.modules, "dora", fake_dora)
    monkeypatch.setitem(sys.modules, "openarm_can", fake_openarm)
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)
    sys.modules.pop("dora_openarm_cell_lifter.main", None)

    return importlib.import_module("dora_openarm_cell_lifter.main")


def _run_with_joystick(main_module, monkeypatch, joystick_y):
    args = types.SimpleNamespace(lead_length=5.0, screw_length=10.0)
    normal_position = math.pi * args.screw_length / args.lead_length
    lifter = FakeLifter(
        calibration_state={"position": 0.0, "torque": 2.0},
        normal_state={"position": normal_position, "torque": 0.0},
    )
    node = FakeNode(
        [
            {"type": "INPUT", "id": "joystick_y", "value": [FakeValue(joystick_y)]},
            {"type": "INPUT", "id": "joystick_y", "value": [FakeValue(joystick_y)]},
        ]
    )
    time_values = iter([100.0, 101.0])

    monkeypatch.setattr(main_module.dora, "Node", lambda: node)
    monkeypatch.setattr(main_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(main_module.time, "time", lambda: next(time_values))

    main_module._dora_main(lifter, args)

    return node.outputs


@pytest.mark.parametrize(
    ("joystick_y", "velocity_sign"),
    [
        (1.0, -1.0),
        (-1.0, 1.0),
    ],
)
def test_joystick_y_produces_expected_elevation_action(
    main_module, monkeypatch, joystick_y, velocity_sign
):
    outputs = _run_with_joystick(main_module, monkeypatch, joystick_y)
    action_outputs = [value for name, value in outputs if name == "elevation_action"]

    assert len(action_outputs) == 1

    expected_elevation = main_module._calc_next_elevation(
        current_elevation=5.0,
        velocity=velocity_sign * main_module.VEL_MAX,
        dt=1.0,
        lead_length=5.0,
    )
    assert action_outputs[0][0] == pytest.approx(expected_elevation)


@pytest.mark.parametrize("joystick_y", [-0.15, 0.0, 0.15])
def test_deadzone_joystick_y_does_not_emit_elevation_action(
    main_module, monkeypatch, joystick_y
):
    outputs = _run_with_joystick(main_module, monkeypatch, joystick_y)

    assert [name for name, _ in outputs] == ["elevation_observation"]
