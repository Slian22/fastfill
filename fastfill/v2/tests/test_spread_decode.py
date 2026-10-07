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
    xy = spread_grid_xy(logits, residuals, 4, size, np.zeros(3), position, ROOM)
    assert np.allclose(xy[0], [1.5, 1.5]) and np.allclose(xy[1], [2.5, 2.5])
    assert np.allclose(xy[2], [1.5, 1.5])  # different height interval: no conflict with the box below


def test_cells_that_push_the_footprint_out_of_the_room_are_skipped():
    logits = _logits(0, 5)[None]  # cell 0 centre (0.5, 0.5) cannot hold a 1.6 m wide box
    xy = spread_grid_xy(logits, torch.zeros(1, 16, 2), 4, np.array([[1.6, 1.6, 1.]]), np.zeros(1), np.zeros((1, 3)), ROOM)
    assert np.allclose(xy[0], [1.5, 1.5])
