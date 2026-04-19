"""Environment-fingerprint capture on SimLog.

Verifies that `capture_environment()` populates the expected keys, that
`SimLog.environment` defaults to an empty dict, that save/load
round-trips preserve it, and that loading a legacy log without the
field leaves `environment` as an empty dict (backward compatibility).
"""
import json

from crawlbot.simulation.logging import SimLog, capture_environment


def test_environment_field_populated_and_roundtrip(tmp_path):
    # 1. capture_environment() returns a dict with every expected key.
    env = capture_environment()
    assert isinstance(env, dict)
    for key in ('python_version', 'mujoco_version', 'pinocchio_version',
                'numpy_version', 'numpy_show_config', 'pip_freeze',
                'env_vars'):
        assert key in env, f'missing key: {key}'

    # Version strings should look like real versions, not error markers.
    for key in ('mujoco_version', 'pinocchio_version', 'numpy_version'):
        assert not env[key].startswith('<error:'), \
            f'{key} probe raised: {env[key]!r}'
        assert env[key] != '<no __version__>', \
            f'{key} has no __version__'

    # show_config / pip_freeze should be non-empty best-effort strings.
    assert env['numpy_show_config'].strip() != ''
    assert env['pip_freeze'].strip() != ''

    # Exactly the three env var names the spec asked for, no more.
    assert set(env['env_vars'].keys()) == {
        'MUJOCO_GL', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'}

    # 2. Fresh SimLog has environment={} (default_factory).
    log = SimLog()
    assert log.environment == {}

    # 3. Save/load round-trips the populated environment dict.
    log.environment = env
    p = tmp_path / 'sim_log.json'
    log.save(str(p))
    restored = SimLog.load(str(p))
    assert restored.environment == env

    # 4. Legacy log (no 'environment' key) loads and defaults to {}.
    legacy_path = tmp_path / 'legacy.json'
    legacy_path.write_text(json.dumps({'t': [0.0, 0.1],
                                       'phase': ['DS', 'SS']}))
    legacy = SimLog.load(str(legacy_path))
    assert legacy.environment == {}
