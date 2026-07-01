import pytest

from ingen_pydev.algo.spatial_index import SpatialIndex2D


def item_ids(items: list) -> set[str]:
    return {item.item_id for item in items}


def test_insert_and_query_nearby_item() -> None:
    index = SpatialIndex2D(cell_size=10.0)

    index.insert("robot_01", 10.0, 10.0, kind="robot")
    index.insert("robot_02", 12.0, 11.0, kind="robot")
    index.insert("alert_01", 100.0, 100.0, kind="alert")

    results = index.query_radius(10.0, 10.0, radius=5.0)

    assert item_ids(results) == {"robot_01", "robot_02"}


def test_query_excludes_items_outside_radius_even_if_in_checked_cells() -> None:
    index = SpatialIndex2D(cell_size=10.0)

    index.insert("near", 0.0, 0.0)
    index.insert("far", 9.0, 9.0)

    results = index.query_radius(0.0, 0.0, radius=5.0)

    assert item_ids(results) == {"near"}


def test_kind_filter_returns_only_matching_items() -> None:
    index = SpatialIndex2D(cell_size=10.0)

    index.insert("robot_01", 0.0, 0.0, kind="robot")
    index.insert("alert_01", 1.0, 1.0, kind="alert")
    index.insert("charger_01", 2.0, 2.0, kind="charger")

    results = index.query_radius(0.0, 0.0, radius=5.0, kind="alert")

    assert item_ids(results) == {"alert_01"}


def test_update_moves_item_between_cells() -> None:
    index = SpatialIndex2D(cell_size=10.0)

    index.insert("robot_01", 1.0, 1.0, kind="robot")

    old_results = index.query_radius(1.0, 1.0, radius=2.0)
    assert item_ids(old_results) == {"robot_01"}

    index.update("robot_01", 50.0, 50.0)

    old_results = index.query_radius(1.0, 1.0, radius=2.0)
    new_results = index.query_radius(50.0, 50.0, radius=2.0)

    assert item_ids(old_results) == set()
    assert item_ids(new_results) == {"robot_01"}


def test_remove_deletes_item_from_index() -> None:
    index = SpatialIndex2D(cell_size=10.0)

    index.insert("robot_01", 1.0, 1.0, kind="robot")
    index.remove("robot_01")

    results = index.query_radius(1.0, 1.0, radius=5.0)

    assert item_ids(results) == set()
    assert len(index) == 0


def test_negative_coordinates_work_correctly() -> None:
    index = SpatialIndex2D(cell_size=10.0)

    index.insert("robot_negative", -5.0, -5.0, kind="robot")
    index.insert("robot_positive", 20.0, 20.0, kind="robot")

    results = index.query_radius(-5.0, -5.0, radius=3.0)

    assert item_ids(results) == {"robot_negative"}


def test_duplicate_insert_raises_error() -> None:
    index = SpatialIndex2D(cell_size=10.0)

    index.insert("robot_01", 0.0, 0.0)

    with pytest.raises(ValueError):
        index.insert("robot_01", 1.0, 1.0)


def test_negative_radius_raises_error() -> None:
    index = SpatialIndex2D(cell_size=10.0)

    with pytest.raises(ValueError):
        index.query_radius(0.0, 0.0, radius=-1.0)


def test_invalid_cell_size_raises_error() -> None:
    with pytest.raises(ValueError):
        SpatialIndex2D(cell_size=0.0)
