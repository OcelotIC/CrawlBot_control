"""COM-GAIN-AUDIT: prove the new gain tests bite on the pre-fix code path.

A test that passes after a fix is worthless unless it FAILS before it. This
re-runs each new assertion against `old(g) = np.diag(g)` — the exact
pre-fix expression at `wholebody_qp.py:902` — and reports which ones catch it.

Injects nothing into the tree; the old semantics is one line here.
"""
import numpy as np

from crawlbot.solvers.wholebody_qp import as_gain_matrix


def old(g, n=3, name=''):
    """The pre-fix expression, verbatim: a bare np.diag on the config field."""
    return np.diag(g)


def check(label, fn, expect_raise):
    """Run a predicate. Returns True if it raised.

    `expect_raise` only selects the wording: on the pre-fix path a raise is the
    test doing its job; on the post-fix path a clean run is.
    """
    try:
        fn()
    except (AssertionError, ValueError, TypeError, IndexError) as exc:
        tag = 'CAUGHT' if expect_raise else 'REGRESS'
        print(f'  {tag:<7} {label:<46} ({type(exc).__name__}: {exc})'
              if not expect_raise else
              f'  {tag:<7} {label:<46} ({type(exc).__name__})')
        return True
    tag = 'BLIND' if expect_raise else 'ok'
    note = ' (test cannot see the defect)' if expect_raise else ''
    print(f'  {tag:<7} {label:<46}{note}')
    return False


def suite(K, expect_raise):
    """The five behavioural predicates, parameterized on the gain builder."""
    results = []

    def t_forms():
        want = 3.0 * np.eye(3)
        for g in (3.0 * np.ones(3), np.diag([3.0, 3.0, 3.0])):
            got = K(g, 3, 'Kp_com')
            assert np.shape(got) == (3, 3), f'shape {np.shape(got)}'
            assert np.allclose(got, want)
    results.append(check('three input forms agree', t_forms, expect_raise))

    def t_rank():
        M = K(np.diag([3.0, 3.0, 5.0]), 3, 'Kp_com')
        assert np.linalg.matrix_rank(M) == 3, f'rank {np.linalg.matrix_rank(M)}'
        out = M @ np.array([0.01, -0.02, 0.03])
        assert np.shape(out) == (3,), f'got {np.shape(out)}'
        assert np.allclose(out, [0.03, -0.06, 0.15]), out
    results.append(check('matrix input is not collapsed', t_rank, expect_raise))

    def t_aniso():
        M = K(np.diag([50.0, 50.0, 100.0]), 3, 'Kp_com')
        out = M @ np.array([0.0, 0.0, 1.0])
        assert np.allclose(out, [0.0, 0.0, 100.0]), out
    results.append(check('anisotropy survives', t_aniso, expect_raise))

    def t_perp():
        M = K(np.diag([3.0, 3.0, 3.0]), 3, 'Kp_com')
        out = M @ np.array([1.0, -1.0, 0.0])
        assert np.linalg.norm(out) > 1e-9, f'no feedback: {out}'
        assert np.allclose(out, [3.0, -3.0, 0.0]), out
    results.append(check('error orthogonal to [1,1,1] is seen', t_perp, expect_raise))

    def t_reject():
        try:
            K(np.ones(4), 3, 'Kp_com')
        except ValueError:
            return
        raise AssertionError('wrong shape (4,) accepted silently')
    results.append(check('bad shapes rejected', t_reject, expect_raise))

    return results


def main():
    print('=== pre-fix path: np.diag(cfg.Kp_com) ===')
    caught = suite(old, expect_raise=True)
    n_caught = sum(caught)

    print('\n=== post-fix path: as_gain_matrix(cfg.Kp_com, 3) ===')
    survived = suite(as_gain_matrix, expect_raise=False)
    n_failed = sum(survived)

    print(f'\npre-fix : {n_caught}/{len(caught)} predicates caught the defect')
    print(f'post-fix: {n_failed}/{len(survived)} predicates failing '
          f'(must be 0)')

    # Quantify the collapse on the canonical gain, isotropic 3.0.
    print('\n=== the collapse, on the canonical gain diag([3,3,3]) ===')
    g = np.diag([3.0, 3.0, 3.0])
    for e, tag in ((np.array([1e-3, 0.0, 0.0]), 'e = x only'),
                   (np.array([1e-3, -1e-3, 0.0]), 'e ⟂ [1,1,1]'),
                   (np.array([1e-3, 1e-3, 1e-3]), 'e ∥ [1,1,1]')):
        a_new = as_gain_matrix(g, 3) @ e
        a_old = np.atleast_1d(old(g) @ e) * np.ones(3)
        print(f'  {tag:<14} intended={np.array2string(a_new * 1e3, precision=3):<26}'
              f' applied={np.array2string(a_old * 1e3, precision=3)}  [mm/s^2]')

    ok = (n_caught == len(caught)) and (n_failed == 0)
    print(f'\nBITE CHECK: {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
