"""Check format and provenance-invariant properties of example COMSOL data."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with (ROOT / path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    config = rows('data/chained_hist/params.csv')
    fields = rows('data/examples/h0000_field_snapshots.csv')
    current = rows('data/examples/h0000_surface_current.csv')

    assert len(config) == 120
    assert len({r['name'] for r in config}) == 120
    assert all(r['out'] == f"data/chained_hist/runs/{r['name']}" for r in config)
    assert len(fields) == 318 * 4
    assert len(current) == 39 * 4
    times = {'0', '0.881', '2', '3.954'}
    assert {r['time_d'] for r in fields} == times
    assert {r['time_d'] for r in current} == times
    for rs in (fields, current):
        assert {r['trajectory_id'] for r in rs} == {'h0000'}
        assert all(float(r['r_m']) >= 0 for r in rs)
    print('public data checks: PASS (120 schedules; 318 field nodes; 39 current nodes)')


if __name__ == '__main__':
    main()
