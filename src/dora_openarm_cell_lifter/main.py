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

"""dora-rs node to control OpenArm Cell Lifter."""

import argparse
import dora
import openarm_can as oa
import math
import os
import time

import pyarrow as pa

# --- Constants ---
VEL_MAX = 30.0
POS_MIN = 0.0
HOLD_POS_VEL = 1.0

# Speed of the move to the start elevation after calibration, and how close
# (rad) counts as arrived, after which the position is held with HOLD_POS_VEL.
START_MOVE_VEL = VEL_MAX / 5.0
START_REACHED_RAD = 0.1

# Used for calibration and homing
CALIB_TORQUE_LIMIT = 0.7

# Threshold for joystick deadzone
JOYSTICK_DEADZONE = 0.15
# Scaling factor for velocity calculation (1.0 - 0.15)
JOYSTICK_RANGE = 0.85

# Torque threshold for anomaly detection (collision / mechanical limit)
TORQUE_LIMIT = 1.5


class PositionUnwrapper:
    """Class to handle wrap-around of multi-turn motor encoders (-4pi to 4pi)."""

    def __init__(self, wrap_range=8.0 * math.pi, wrap_threshold=4.0 * math.pi):
        """Initialize the tracker for wrapped encoder positions."""
        self._wrap_range = wrap_range
        self._wrap_threshold = wrap_threshold
        self._prev_raw = None
        self._continuous_pos = 0.0

    def update(self, current_raw):
        """Update the continuous position estimate from the raw encoder value."""
        if self._prev_raw is None:
            self._prev_raw = current_raw
            self._continuous_pos = current_raw
            return self._continuous_pos

        diff = current_raw - self._prev_raw

        # Unwrap logic for 8pi range
        if diff > self._wrap_threshold:
            diff -= self._wrap_range
        elif diff < -self._wrap_threshold:
            diff += self._wrap_range

        self._continuous_pos += diff
        self._prev_raw = current_raw
        return self._continuous_pos


def _start_elevation(value, screw_length):
    """Return the start elevation (mm) clamped to the stroke, or None if off."""
    value = (value or "").strip().lower()
    if value in ("", "off"):
        return None
    return max(0.0, min(screw_length, float(value)))


def _reached(position, target):
    """Return whether the motor position (rad) is at the target."""
    return abs(position - target) < START_REACHED_RAD


def _calc_next_elevation(current_elevation, velocity, dt, lead_length):
    return current_elevation + velocity * dt * lead_length / (2.0 * math.pi)


def _calc_elevation_action_from_joystick(
    current_elevation, joystick_y, dt, lead_length, speed_factor=1.0
):
    if joystick_y > JOYSTICK_DEADZONE:
        applied_vel = VEL_MAX * (abs(joystick_y - JOYSTICK_DEADZONE) / JOYSTICK_RANGE)
        applied_vel *= speed_factor
        return _calc_next_elevation(
            current_elevation, applied_vel, dt, lead_length
        ), applied_vel

    elif joystick_y < -JOYSTICK_DEADZONE:
        applied_vel = VEL_MAX * (abs(joystick_y + JOYSTICK_DEADZONE) / JOYSTICK_RANGE)
        applied_vel *= speed_factor
        return _calc_next_elevation(
            current_elevation, -applied_vel, dt, lead_length
        ), applied_vel

    else:
        return None, 0.0


def _dora_main(lifter, args):
    pos_max = (args.screw_length / args.lead_length) * 2.0 * math.pi
    slow_margin = pos_max * 0.05

    node = dora.Node()

    lifter_pos = 0.0
    lifter_tau = 0.0

    joystick_y = 0.0

    for motor in lifter.get_arm().get_motors():
        lifter_pos = motor.get_position()
        lifter_tau = motor.get_torque()

    unwrapper = PositionUnwrapper()

    calibrated = False
    offset_pos = 0.0

    # State management variables
    jammed_direction = None
    hold_pos = 0.0
    hold_elevation = 0.0
    is_stopping = False
    # Driving to the start elevation after calibration (faster than a hold).
    moving_to_start = False

    prev_time = time.time()

    calib_tick = 0
    if args.debug:
        print(
            f"[LIFTER] Starting event loop. Initial motor state: pos={lifter_pos:.4f} rad, tau={lifter_tau:.4f} Nm",
            flush=True,
        )
        print(
            f"[LIFTER] pos_max={pos_max:.4f} rad, torque_limit=1.0 Nm (calibration)",
            flush=True,
        )

    # For calibration: give runway for torque sensing
    lifter.get_arm().posvel_control_all(
        [oa.PosVelParam(q=POS_MIN + 2.0, dq=VEL_MAX / 5.0)]
    )
    time.sleep(1.0)

    for event in node:
        event_type = event["type"]
        if event_type != "INPUT":
            if args.debug:
                print(f"[LIFTER] Non-INPUT event: type={event_type}", flush=True)
            continue

        event_id = event.get("id")

        value = event.get("value")
        if value is None:
            if args.debug:
                print(f"[LIFTER] Event '{event_id}' has no value, skipping", flush=True)
            continue

        if event_id == "joystick_y":
            py_value = value[0].as_py()
            joystick_y = py_value
        elif event_id == "command":
            command = event["value"][0].as_py()
            if command == "lifter-up":
                joystick_y = 0.5
            elif command == "lifter-down":
                joystick_y = -0.5
            else:
                joystick_y = 0
        elif event_id == "move_elevation":
            pass  # handled in normal operation phase
        elif event_id == "tick":
            pass
        else:
            if args.debug:
                print(f"[LIFTER] Unknown event_id='{event_id}', skipping", flush=True)
            continue

        # --- Calibration Phase ---
        if not calibrated:
            lifter.get_arm().posvel_control_all(
                [oa.PosVelParam(q=POS_MIN - 1000.0, dq=VEL_MAX / 5.0)]
            )

            lifter.recv_all()

            for motor in lifter.get_arm().get_motors():
                lifter_pos = motor.get_position()
                lifter_tau = motor.get_torque()

            calib_tick += 1
            if args.debug:
                print(
                    f"[CALIB tick={calib_tick}] event={event_id}  pos={lifter_pos:.4f} rad  tau={lifter_tau:.4f} Nm  (threshold={CALIB_TORQUE_LIMIT})",
                    flush=True,
                )

            if abs(lifter_tau) > CALIB_TORQUE_LIMIT:
                offset_pos = lifter_pos
                calibrated = True
                # With a start elevation, move there and hold it, like the
                # arms' start pose, until the first lifter command. Without
                # one, hold where calibration ended (the bottom).
                start_elevation = _start_elevation(
                    args.start_elevation, args.screw_length
                )
                if start_elevation is not None:
                    hold_elevation = start_elevation
                    hold_pos = (
                        start_elevation / args.lead_length * 2.0 * math.pi + offset_pos
                    )
                    is_stopping = True
                    moving_to_start = True
                if args.debug:
                    print(
                        f"[CALIB] Done! offset_pos={offset_pos:.4f} rad after {calib_tick} ticks. "
                        f"Start elevation: {start_elevation} mm.",
                        flush=True,
                    )
            continue

        # --- Normal Operation Phase ---
        now = time.time()
        dt = now - prev_time
        prev_time = now

        lifter.recv_all()

        for motor in lifter.get_arm().get_motors():
            lifter_pos = motor.get_position()
            lifter_tau = motor.get_torque()
        obs_position = unwrapper.update(lifter_pos) - offset_pos
        # obs: elevation(mm)
        obs_elevation = obs_position / (2.0 * math.pi) * args.lead_length
        node.send_output(
            "elevation_observation", pa.array([obs_elevation], type=pa.float32())
        )

        # Calculate distance to the nearest physical limit (lower 0.0 or upper pos_max)
        distance_to_min = obs_position - 0.0
        distance_to_max = pos_max - obs_position
        distance_to_edge = min(distance_to_min, distance_to_max)

        # Calculate the linear deceleration factor
        if distance_to_edge < slow_margin:
            # Clamp to prevent negative values due to slight sensor noise
            clamped_distance = max(0.0, distance_to_edge)
            # Linear interpolation: 0.25x at the edge, 1.0x at the slow_margin boundary
            speed_factor = 0.25 + 0.75 * (clamped_distance / slow_margin)
        else:
            speed_factor = 1.0

        # Any explicit move ends the drive to the start elevation.
        if event_id == "move_elevation" or abs(joystick_y) > JOYSTICK_DEADZONE:
            moving_to_start = False

        # --- Elevation (mm) ---
        if event_id == "move_elevation":
            elevation = max(0.0, min(args.screw_length, value[0].as_py()))
            target_position = elevation / args.lead_length * 2.0 * math.pi + offset_pos
            is_stopping = False

            node.send_output(
                "elevation_action",
                pa.array([elevation], type=pa.float32()),
            )

            lifter.get_arm().posvel_control_all(
                [oa.PosVelParam(q=target_position, dq=VEL_MAX * speed_factor)]
            )
            continue

        # --- 1. Jam detection and direction memorization ---
        if abs(lifter_tau) > TORQUE_LIMIT:
            if jammed_direction is None:
                # Memorize the exact position at impact to prevent hunting
                hold_pos = obs_position + offset_pos  # Use the unwrapper position
                hold_elevation = obs_elevation  # obs elevation is calculated from the unwrapper lifter_pos(hold_pos)

                if joystick_y > JOYSTICK_DEADZONE:
                    jammed_direction = "UP"
                elif joystick_y < -JOYSTICK_DEADZONE:
                    jammed_direction = "DOWN"
                else:
                    jammed_direction = "UNKNOWN"
        else:
            # Do not release the lock if the joystick is held in the same direction,
            # even if the torque drops to a normal value
            if jammed_direction == "UP" and joystick_y <= JOYSTICK_DEADZONE:
                jammed_direction = None
            elif jammed_direction == "DOWN" and joystick_y >= -JOYSTICK_DEADZONE:
                jammed_direction = None
            elif jammed_direction == "UNKNOWN":
                jammed_direction = None

        # --- 2. Action determination (One-way block, reverse allow, linear deceleration at limits) ---

        # UP operation
        if joystick_y > JOYSTICK_DEADZONE:
            is_stopping = False
            if jammed_direction in ["UP", "UNKNOWN"]:
                # Hold silently at the memorized position while jammed in the UP direction
                node.send_output(
                    "elevation_action",
                    pa.array([hold_elevation], type=pa.float32()),
                )
                lifter.get_arm().posvel_control_all(
                    [oa.PosVelParam(q=hold_pos, dq=0.0)]
                )

            else:
                action_elevation, applied_vel = _calc_elevation_action_from_joystick(
                    obs_elevation,
                    joystick_y,
                    dt,
                    args.lead_length,
                    speed_factor,
                )

                node.send_output(
                    "elevation_action",
                    pa.array([action_elevation], type=pa.float32()),
                )

                # Drive toward the top stroke limit (positive joystick_y = up)
                lifter.get_arm().posvel_control_all(
                    [oa.PosVelParam(q=pos_max + offset_pos, dq=applied_vel)]
                )

        # DOWN operation
        elif joystick_y < -JOYSTICK_DEADZONE:
            is_stopping = False
            if jammed_direction in ["DOWN", "UNKNOWN"]:
                # Hold silently at the memorized position while jammed in the DOWN direction

                node.send_output(
                    "elevation_action",
                    pa.array([hold_elevation], type=pa.float32()),
                )
                lifter.get_arm().posvel_control_all(
                    [oa.PosVelParam(q=hold_pos, dq=0.0)]
                )
            else:
                action_elevation, applied_vel = _calc_elevation_action_from_joystick(
                    obs_elevation,
                    joystick_y,
                    dt,
                    args.lead_length,
                    speed_factor,
                )

                node.send_output(
                    "elevation_action",
                    pa.array([action_elevation], type=pa.float32()),
                )

                # Drive toward the bottom stroke limit (negative joystick_y = down)
                lifter.get_arm().posvel_control_all(
                    [oa.PosVelParam(q=offset_pos, dq=applied_vel)]
                )

        # STOP (Within deadzone)
        else:
            if not is_stopping:
                # Memorize the exact position at the moment the joystick is released
                hold_pos = obs_position + offset_pos  # Use the unwrapper position
                hold_elevation = obs_elevation  # obs elevation is calculated from the unwrapper position(hold_pos)
                is_stopping = True

            node.send_output(
                "elevation_action",
                pa.array([hold_elevation], type=pa.float32()),
            )
            if moving_to_start and _reached(obs_position + offset_pos, hold_pos):
                moving_to_start = False
            # Hold position with a small velocity to maintain stiffness, preventing drift due to gravity or disturbances
            lifter.get_arm().posvel_control_all(
                [
                    oa.PosVelParam(
                        q=hold_pos,
                        dq=START_MOVE_VEL if moving_to_start else HOLD_POS_VEL,
                    )
                ]
            )


def main():
    """Control the OpenArm Lifter using joystick inputs.

    observation:
    elevation (mm) from the bottom, converted from motor angle (rad) using the lead length.

    action:
    Estimated next elevation (mm), calculated from observation and `applied_vel`.
    `applied_vel` is reduced linearly near the stroke limits (within `SLOW_MARGIN`) to avoid collision.
    `action` is only output when the joystick is operated.
    """
    parser = argparse.ArgumentParser(description="Control the OpenArm Lifter")
    parser.add_argument(
        "--can-interface",
        default=os.getenv("CAN_INTERFACE", "can2"),
        help="The CAN interface name",
    )
    parser.add_argument(
        "--lead-length",
        default=float(os.getenv("LEAD_LENGTH", 5.0)),
        help="Lead screw lead length in mm/rev",
        type=float,
    )
    parser.add_argument(
        "--screw-length",
        default=float(os.getenv("SCREW_LENGTH", 300.0)),
        help="Lead screw stroke length in mm",
        type=float,
    )
    parser.add_argument(
        "--start-elevation",
        default=os.getenv("START_ELEVATION", "off"),
        help=(
            "Elevation in mm to move to and hold after calibration (0 = bottom), "
            "or 'off' to stay where calibration ends"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug prints for calibration and event flow",
    )
    args = parser.parse_args()
    try:
        _start_elevation(args.start_elevation, args.screw_length)
    except ValueError:
        parser.error(
            f"--start-elevation: not a number or 'off': {args.start_elevation}"
        )

    lifter = oa.OpenArm(args.can_interface, enable_fd=True)

    motor_types = [oa.MotorType.DM4310]
    send_ids = [0x0A]
    recv_ids = [0x1A]
    control_modes = [oa.ControlMode.POS_VEL]

    lifter.init_arm_motors(motor_types, send_ids, recv_ids, control_modes)

    # The motor starts in whatever mode was last saved to its flash (MIT from
    # the factory) and ignores position/velocity commands in any other mode,
    # leaving the lifter unable to move or calibrate. Set the mode on every
    # start, before enabling; it is not saved to flash.
    lifter.set_callback_mode_all(oa.CallbackMode.IGNORE)
    lifter.get_arm().set_control_mode_all(oa.ControlMode.POS_VEL)
    lifter.recv_all()

    lifter.set_callback_mode_all(oa.CallbackMode.STATE)

    lifter.enable_all()
    lifter.recv_all()

    _dora_main(lifter, args)

    # Move the lifter down to the mechanical stop before disabling
    homing_tick = 0
    while True:
        lifter.get_arm().posvel_control_all(
            [oa.PosVelParam(q=POS_MIN - 1000.0, dq=VEL_MAX / 5.0)]
        )

        lifter.recv_all()

        for motor in lifter.get_arm().get_motors():
            lifter_tau = motor.get_torque()
            lifter_pos = motor.get_position()
            homing_tick += 1
            if args.debug:
                print(
                    f"[HOMING tick={homing_tick}]   pos={lifter_pos:.4f} rad  tau={lifter_tau:.4f} Nm  (threshold={CALIB_TORQUE_LIMIT})",
                    flush=True,
                )
        if abs(lifter_tau) > CALIB_TORQUE_LIMIT:
            break
        # stops the shutdown homing when abs(lifter_tau) > CALIB_TORQUE_LIMIT

    # Disable motors for safety after exiting the loop
    lifter.disable_all()


if __name__ == "__main__":
    main()
