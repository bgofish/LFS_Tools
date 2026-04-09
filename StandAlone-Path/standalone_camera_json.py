#!/usr/bin/env python3
"""
Standalone Camera JSON Generator
Generates camera path JSON files in SuperSplat keyframe format (version 3).

Output format:
{
  "keyframes": [
    {
      "easing": 0,
      "focal_length_mm": 35.0,
      "position": [x, y, z],
      "rotation": [qw, qx, qy, qz],
      "time": 0.0
    },
    ...
  ],
  "version": 3
}
"""

import json
import math


def focal_length_to_fov(focal_length_mm: float, sensor_size_mm: float = 32.0) -> float:
    """Convert focal length (mm) to horizontal field of view (degrees)."""
    return 2.0 * math.degrees(math.atan(sensor_size_mm / (2.0 * focal_length_mm)))


def normalize_vector(v):
    """Normalize a 3D vector."""
    length = math.sqrt(sum(c * c for c in v))
    if length < 1e-10:
        return (0.0, 0.0, 1.0)
    return tuple(c / length for c in v)


def look_at_quaternion(position, target):
    """
    Compute a quaternion that rotates from the default camera orientation
    (looking down -Z, Y up) to look from `position` toward `target`.

    Returns (qw, qx, qy, qz).
    """
    # Forward vector (from position toward target)
    fx = target[0] - position[0]
    fy = target[1] - position[1]
    fz = target[2] - position[2]
    forward = normalize_vector((fx, fy, fz))

    # Choose a world-up vector; if forward is nearly parallel to Y, use Z instead
    if abs(forward[1]) > 0.99:
        world_up = (0.0, 0.0, -1.0 if forward[1] > 0 else 1.0)
    else:
        world_up = (0.0, 1.0, 0.0)

    # Right vector: world_up x forward  (SuperSplat convention)
    right = normalize_vector((
        world_up[1] * forward[2] - world_up[2] * forward[1],
        world_up[2] * forward[0] - world_up[0] * forward[2],
        world_up[0] * forward[1] - world_up[1] * forward[0],
    ))

    # Recompute up orthogonally: forward x right
    up = (
        forward[1] * right[2] - forward[2] * right[1],
        forward[2] * right[0] - forward[0] * right[2],
        forward[0] * right[1] - forward[1] * right[0],
    )

    # Build rotation matrix with +Z = forward (SuperSplat convention)
    # Column 0 = right, Column 1 = up, Column 2 = +forward
    m00, m10, m20 = right
    m01, m11, m21 = up
    m02, m12, m22 = forward[0], forward[1], forward[2]

    # Convert 3x3 rotation matrix to quaternion
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m21 - m12) * s
        qy = (m02 - m20) * s
        qz = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    # Normalise
    length = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return (qw / length, qx / length, qy / length, qz / length)


class StandaloneCameraGenerator:
    """Generates camera animations and writes them in SuperSplat keyframe JSON format."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_camera_animation(
        self,
        animation_type: str = "circular",
        direction: str = "clockwise",
        center: tuple = (0.0, 0.0, 0.0),
        target: tuple = None,
        target_distance: float = None,
        radius: float = 10.0,
        start_radius: float = 5.0,
        end_radius: float = 15.0,
        start_height: float = 0.0,
        end_height: float = 10.0,
        spiral_loops: float = 2.0,
        frames: int = 180,
        fps: int = 24,
        focal_length: float = 35.0,
        sensor_size: float = 32.0,
        convert_coords: bool = False,
        precision: int = 6,
        keyframe_step: int = 1,
        spiral_follow_y: bool = False,
        spiral_y_offset: float = 0.0,
    ) -> dict:
        """
        Generate camera animation data in SuperSplat keyframe format.

        Parameters
        ----------
        animation_type : "circular" or "spiral"
        direction      : "clockwise" or "counterclockwise"
        center         : (x, y, z) centre of the orbit/spiral
        target         : (x, y, z) point the camera looks at; if None,
                         auto-calculated from target_distance
        target_distance: used when target is None
        radius         : orbit radius for circular animation
        start_radius   : starting radius for spiral animation
        end_radius     : ending radius for spiral animation
        start_height   : starting height (Y axis) for spiral
        end_height     : ending height (Y axis) for spiral
        spiral_loops   : number of full rotations in spiral
        frames         : total number of frames
        fps            : frames per second (used to compute keyframe time values)
        focal_length   : camera focal length in mm
        sensor_size    : sensor width in mm (used for FOV display only)
        convert_coords : if True, swaps Y/Z axes (Z-up → Y-up)
        precision      : decimal places for floating-point values
        keyframe_step  : emit a keyframe every N frames
        spiral_follow_y: if True (spiral only), the look-target Y tracks the
                         camera's current height instead of staying fixed
        spiral_y_offset: added to the look-target Y when spiral_follow_y is on
                         (positive = look above camera height, negative = below)

        Returns
        -------
        dict  ready to be serialised as JSON
        """
        keyframes = []
        total_duration = frames / fps  # seconds

        # Angle multiplier for direction
        angle_sign = -1.0 if direction == "clockwise" else 1.0

        for frame_idx in range(0, frames, keyframe_step):
            t_norm = frame_idx / max(frames - 1, 1)  # 0.0 → 1.0
            time_sec = round(frame_idx / fps, precision)

            angle = angle_sign * 2.0 * math.pi * t_norm

            # --- Compute camera position ---
            if animation_type == "circular":
                r = radius
                height = center[1]
                angle_full = angle  # one full revolution
            else:
                # Spiral: interpolate radius and height, multiple loops
                r = start_radius + (end_radius - start_radius) * t_norm
                height = center[1] + start_height + (end_height - start_height) * t_norm
                angle_full = angle_sign * 2.0 * math.pi * spiral_loops * t_norm

            px = center[0] + r * math.sin(angle_full)
            py = height
            pz = center[2] + r * math.cos(angle_full)

            position = [px, py, pz]

            # --- Compute look-at target ---
            if target is not None:
                look_target = list(target)
            elif target_distance is not None:
                # Auto: look toward the centre at the given distance
                look_target = [
                    center[0],
                    center[1] if animation_type == "circular" else height,
                    center[2],
                ]
            else:
                look_target = list(center)

            # Spiral follow-Y: override the target's Y to track camera height
            if animation_type == "spiral" and spiral_follow_y:
                look_target[1] = height + spiral_y_offset

            # --- Optional coordinate conversion (Z-up → Y-up) ---
            if convert_coords:
                position = [position[0], position[2], -position[1]]
                look_target = [look_target[0], look_target[2], -look_target[1]]

            # --- Quaternion ---
            qw, qx, qy, qz = look_at_quaternion(position, look_target)

            def r_val(v):
                return round(v, precision)

            keyframes.append({
                "easing": 0,
                "focal_length_mm": round(focal_length, precision),
                "position": [r_val(position[0]), r_val(position[1]), r_val(position[2])],
                "rotation": [r_val(qw), r_val(qx), r_val(qy), r_val(qz)],
                "time": r_val(time_sec),
            })

        return {
            "keyframes": keyframes,
            "version": 3,
        }

    def save_json(self, data: dict, output_path: str) -> None:
        """Write the animation data to a JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
