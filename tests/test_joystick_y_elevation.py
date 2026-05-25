# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import sys
import types
from pathlib import Path

import pytest


sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)


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


@pytest.mark.parametrize(
    ("joystick_y", "expected_velocity"),
    [
        (1.0, -30.0),
        (-1.0, 30.0),
    ],
)
def test_joystick_y_produces_expected_elevation_action(
    main_module, joystick_y, expected_velocity
):
    action_elevation, applied_vel = main_module._calc_elevation_action_from_joystick(
        current_elevation=5.0,
        joystick_y=joystick_y,
        dt=1.0,
        lead_length=5.0,
    )

    expected_elevation = main_module._calc_next_elevation(
        current_elevation=5.0,
        velocity=expected_velocity,
        dt=1.0,
        lead_length=5.0,
    )
    assert action_elevation == pytest.approx(expected_elevation)
    assert applied_vel == pytest.approx(main_module.VEL_MAX)


@pytest.mark.parametrize("joystick_y", [-0.15, 0.0, 0.15])
def test_deadzone_joystick_y_does_not_produce_elevation_action(main_module, joystick_y):
    action_elevation, applied_vel = main_module._calc_elevation_action_from_joystick(
        current_elevation=5.0,
        joystick_y=joystick_y,
        dt=1.0,
        lead_length=5.0,
    )

    assert action_elevation is None
    assert applied_vel == 0.0
