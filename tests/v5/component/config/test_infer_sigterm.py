"""defect49: interrupted infer must exit and release the GPU, never hang on interpreter shutdown.

PLAN (case | input | expected | kills):
 1 sigterm | real SIGTERM mid-run + live non-daemon sleeper | exit 143 fast, cleanup ran | threading._shutdown join hang
 2 sigint | real SIGINT, same shape | exit 130, cleanup ran | wrong signal-to-code mapping
 3 clean | no signal, normal return | exit 0, no hard-exit marker | over-correction: os._exit on the success path
 4 dfl-restored | catch first SystemExit, inspect handler | SIGTERM back to SIG_DFL | second-signal escalation broken

Oracle = real subprocess exit codes + cleanup marker file on disk. The non-daemon sleeper thread
reproduces the tqdm_monitor failure surface: without the wrapper's hard exit the interpreter joins
it forever. Reverse-verify: drop os._exit from infer_main's finally -> case 1 times out (RED).
"""

import os
import subprocess
import sys
import textwrap

import pytest

_CHILD = textwrap.dedent('''
    import os, signal, sys, threading, time
    from swift.dev.cli import infer as infer_mod

    mode, marker = sys.argv[1], sys.argv[2]

    def fake_main(argv=None):
        if mode in ('sigterm', 'sigint'):
            t = threading.Thread(target=lambda: time.sleep(3600))
            t.daemon = False
            t.start()
            try:
                os.kill(os.getpid(), signal.SIGTERM if mode == 'sigterm' else signal.SIGINT)
                time.sleep(5)
            finally:
                open(marker, 'w').write('cleaned')
        elif mode == 'dfl':
            try:
                os.kill(os.getpid(), signal.SIGTERM)
            except SystemExit:
                ok = signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
                open(marker, 'w').write('dfl-ok' if ok else 'dfl-broken')
                raise
        return None

    infer_mod._infer_main = fake_main
    infer_mod.infer_main()
''')


def _run(tmp_path, mode):
    script = tmp_path / 'child.py'
    script.write_text(_CHILD)
    marker = tmp_path / 'marker'
    env = dict(os.environ, USE_SWIFT_V5='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    env['PYTHONPATH'] = os.pathsep.join([os.path.join(repo, 'twinkle', 'src'), repo])
    return subprocess.run([sys.executable, str(script), mode, str(marker)],
                          capture_output=True, timeout=60, env=env), marker


@pytest.mark.parametrize('mode,code', [('sigterm', 143), ('sigint', 130)])
def test_signal_exits_fast_with_cleanup(tmp_path, mode, code):
    proc, marker = _run(tmp_path, mode)
    assert proc.returncode == code, proc.stderr.decode()[-800:]
    assert marker.read_text() == 'cleaned'


def test_clean_return_not_hard_exited(tmp_path):
    proc, marker = _run(tmp_path, 'clean')
    assert proc.returncode == 0, proc.stderr.decode()[-800:]
    assert not marker.exists()


def test_second_signal_restores_default(tmp_path):
    proc, marker = _run(tmp_path, 'dfl')
    assert proc.returncode == 143, proc.stderr.decode()[-800:]
    assert marker.read_text() == 'dfl-ok'
