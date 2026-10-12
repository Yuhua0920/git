"""Actual-algorithm, synthetic-only theory verification; no formal-output reader."""
import numpy as np
import pytest
from scripts.validate_sp_u2a import (SENTINELS, family, expand, maps, realify, output_guard)


@pytest.mark.parametrize('name', list(SENTINELS))
def test_independent_theory_sentinel(name):
    assert SENTINELS[name]()


def test_noisy_input_design_and_residual_identity():
    cv, pilots, _ = family()
    u, tp, tm = maps(cv)
    base = pilots[:1]
    noise = np.cos(np.arange(64) / 5) + 1j * np.sin(np.arange(64) / 7)
    before = expand(cv, base)
    after = expand(cv, base + noise)
    np.testing.assert_allclose(after[0] - before[0], realify(np.column_stack([tp @ noise, tm @ noise])), atol=1e-13)
    np.testing.assert_allclose(after[1] - before[1], realify(u @ noise), atol=1e-13)
    c = np.array([.4, -.3])
    np.testing.assert_allclose(after[1] - after[0] @ c,
                               realify((u - c[0] * tp - c[1] * tm) @ (base[0] + noise - cv.source_mean_p)), atol=1e-13)


def test_old_scientific_output_firewall_rejects_before_io():
    for path in ('outputs/t20b_formal_nmse.csv', 'outputs/t18a_validation.csv', 'outputs/t20c_result.json'):
        with pytest.raises(PermissionError):
            output_guard('open', (path, 'r', 0))
    output_guard('open', ('outputs/sp_u2a_theory_validation.json', 'r', 0))


def test_geometry_condition_number_alone_does_not_control_scale():
    g = np.array([[1., 0.], [0., 1.], [0., 0.]])
    weak = g * 1e-6
    assert np.linalg.cond(g.T @ g) == np.linalg.cond(weak.T @ weak)
    np.testing.assert_allclose(np.linalg.eigvalsh(weak.T @ weak), np.linalg.eigvalsh(g.T @ g) * 1e-12)
    z = np.array([1., 1., 0.])
    np.testing.assert_allclose(np.linalg.lstsq(weak, z, rcond=None)[0],
                               np.linalg.lstsq(g, z, rcond=None)[0] * 1e6)


def test_whitespace_audit_preserves_author_source_and_rejects_own_text(tmp_path):
    from scripts.finalize_sp_u2a import whitespace_issues
    names = ['outputs/sp_u2a_sources/README.md.txt', 'docs/sp_u2a_local.md', 'outputs/sp_u2a_failure_execution.txt']
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'example  \n')
    authored, original = whitespace_issues(names, tmp_path)
    assert authored == [dict(path=names[1], line=1)]
    assert original == [dict(path=names[0], line=1), dict(path=names[2], line=1)]
    for name in names:
        assert (tmp_path / name).read_bytes() == b'example  \n'
