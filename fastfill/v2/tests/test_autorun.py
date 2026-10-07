"""Autopilot decisions: selection score and the rescaled next-run configuration."""
import json

from fastfill.v2.autorun import next_config, score


def _report(size, yaw, pos, central=.17, wall=.66, overlap=.01, out=0.):
    collapse = lambda c, w, o, r: {"central_quarter_fraction": c, "mean_nearest_wall_distance_m": w,
                                   "bev_overlap_rate_iou_gt_0.3": o, "out_of_room_fraction": r}
    return {"model": {"reference": {"log_size_error": {"mean": size}, "yaw_error_rad": {"mean": yaw},
                                    "bottom_center_error_m": {"mean": pos}}},
            "baselines": {"category_median_size": {"log_size_error": {"mean": .4}}, "uniform_yaw": {"yaw_error_rad": {"mean": .7}},
                          "room_center_position": {"bottom_center_error_m": {"mean": 2.5}}},
            "collapse": {"predicted": collapse(central, wall, overlap, out), "ground_truth": collapse(.17, .66, .01, 0.)}}


def test_score_prefers_accurate_layouts_and_punishes_collapse_or_leaving_the_room():
    good, worse = score(_report(.3, .6, 2.0)), score(_report(.35, .6, 2.0))
    assert good < worse
    assert score(_report(.3, .6, 2.0, central=.9, wall=2.0)) > good + .7          # everything in the centre
    assert score(_report(.3, .6, 2.0, overlap=.2)) > good + .85                    # stacked
    assert score(_report(.3, .6, 2.0, out=.5)) > good + 2                          # pushed out of the room


def test_next_config_keeps_the_model_and_loss_and_rescales_to_seven_gpus():
    config = {"model": {"position_head": "grid_residual"}, "loss": {"position_cell": .5},
              "training": {"steps": 3887, "batch_size": 1, "gradient_accumulation_steps": 32, "resume": "x",
                           "checkpoint_every": 500, "validate_every": 500},
              "optimizer": {"warmup_steps": 117}}
    out = next_config(config, world_size=7, train_rows=124584, epochs=5)
    assert out["model"] == config["model"] and out["loss"] == config["loss"]
    t = out["training"]
    assert t["gradient_accumulation_steps"] == 14 and t["batch_size"] == 1 and t["resume"] is None
    assert t["steps"] == -(-5 * 124584 // (7 * 14)) and out["optimizer"]["warmup_steps"] == round(.03 * t["steps"])
    assert json.loads(json.dumps(config))["training"]["steps"] == 3887  # input untouched
