"""Collision-aware decoding of the grid position head spreads identical requests instead of stacking them."""
import numpy as np
import torch

from fastfill.v2.evaluate import spread_grid_xy

ROOM = {"origin": [0., 0., 0.], "scale": [4., 4., 3.], "bounds": (np.zeros(2), np.full(2, 4.)), "floor_z": 0.}


def _logits(*cells, grid=4):
    row = torch.full((grid * grid,), -5.)
    for rank, cell in enumerate(cells):
        row[cell] = 5. - rank
    return row


def test_identical_requests_take_distinct_cells_and_raised_items_may_rest_above():
    logits = torch.stack((_logits(5, 10), _logits(5, 10), _logits(5, 10)))
    residuals = torch.zeros(3, 16, 2)
    size = np.array([[1., 1., 1.], [1., 1., 1.], [.2, .2, .2]])
    position = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]])  # third item sits on top of a 1 m box
    xy, _ = spread_grid_xy(logits, residuals, 4, size, np.zeros(3), position, ROOM)
    assert np.allclose(xy[0], [1.5, 1.5]) and np.allclose(xy[1], [2.5, 2.5])
    assert np.allclose(xy[2], [1.5, 1.5])  # different height interval: no conflict with the box below


def test_cells_that_push_the_footprint_out_of_the_room_are_skipped():
    logits = _logits(0, 5)[None]  # cell 0 centre (0.5, 0.5) cannot hold a 1.6 m wide box
    xy, _ = spread_grid_xy(logits, torch.zeros(1, 16, 2), 4, np.array([[1.6, 1.6, 1.]]), np.zeros(1), np.zeros((1, 3)), ROOM)
    assert np.allclose(xy[0], [1.5, 1.5])


def test_wall_adjacent_furniture_turns_its_back_to_the_wall_within_the_models_own_yaw_bins():
    # a 2 x 1 m bed whose argmax cell sits against the x = 0 wall; the yaw head slightly prefers facing the
    # wall (pi) over facing into the room (0): decoding keeps the back to the wall and re-checks the footprint
    logits = _logits(1, 9)[None]  # cell 1 = (0, 1): x centre 0.5 m, footprint touches x = 0 when depth is 1 m
    yaw_logits = np.full((1, 12), -3.); yaw_logits[0, 6], yaw_logits[0, 0] = 2., 1.  # bin 6 = pi, bin 0 = 0
    size = np.array([[1., 2., .6]])  # depth (local x) 1 m along the facing direction, width 2 m
    xy, yaw = spread_grid_xy(logits, torch.zeros(1, 16, 2), 4, size, np.array([np.pi]), np.zeros((1, 3)), ROOM,
                             yaw_logits=yaw_logits, yaw_residuals=np.zeros((1, 12)))
    assert np.allclose(xy[0], [.5, 1.5]) and abs(yaw[0]) < 1e-9


def test_without_yaw_logits_the_model_yaw_is_kept():
    _, yaw = spread_grid_xy(_logits(1)[None], torch.zeros(1, 16, 2), 4, np.array([[1., 1., .5]]), np.array([2.]),
                            np.zeros((1, 3)), ROOM)
    assert yaw[0] == 2.
