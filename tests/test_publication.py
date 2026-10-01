"""Protect publication data coverage and numeric evidence during regeneration."""

from collections import Counter
from pathlib import Path

import pytest

from pitchbench.analysis import publication


@pytest.fixture(autouse=True)
def evidence_repository(monkeypatch):
    monkeypatch.setattr(publication, 'REPO', Path(__file__).resolve().parents[1])


def test_a1_confusion_counts_account_for_every_saved_answer():
    matrices, coverage, sparse, sources = publication.a1_evidence()
    assert len(sources) == 6
    assert len(matrices) == len(coverage) == 18
    counts = Counter()
    for row in sparse:
        counts[row['model'], row['format']] += row['count']
    for row in coverage:
        key = row['model'], row['format']
        assert row['n_total'] == 1159
        assert row['n_total'] == row['n_in_range'] + row['n_outside_range'] + row['n_invalid']
        assert counts[key] == row['n_in_range']
        totals = matrices[key].sum(axis=0)
        assert all(value == pytest.approx(0) or value == pytest.approx(1) for value in totals)


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'invalid'])
def test_publication_rejects_incomplete_or_corrupted_plot_data(corruption):
    rows = publication.read_csv(publication.REPO / 'paper/figures-and-tables/accuracies_by_pitch.csv')
    if corruption == 'missing':
        rows.pop()
    elif corruption == 'duplicate':
        rows.append(rows[0].copy())
    else:
        rows[0]['accuracy'] = 'nan'
    with pytest.raises(ValueError):
        publication.check_aggregate(rows, ('midi', 'spn'), range(48, 73))


def test_pyramid_fonts_are_available_without_system_font_installation():
    from matplotlib.ft2font import FT2Font
    from pitchbench.analysis import pyramid

    fonts = Path(pyramid.__file__).with_name('fonts')
    for weight in ('Regular', 'ExtraBold'):
        assert FT2Font(str(fonts / f'HankenGrotesk-{weight}.ttf')).family_name == 'Hanken Grotesk'
