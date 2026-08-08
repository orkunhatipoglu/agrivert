"""grpc must not be loaded in the Celery parent process.

Celery's `include=` imports app.worker.tasks in the parent, which then forks
its pool children. grpc is not fork-safe: a child forked from a parent that
already imported grpc inherits a dead c-ares resolver, and every Firestore
call then burns its full 60s/300s retry deadline before failing with
"503 errors resolving firestore.googleapis.com: Could not contact DNS
servers". Tasks look like they hang forever, while the API — which never
forks — reaches Firestore fine.

Nothing about that failure points at an import statement, so this test exists
to fail loudly the moment someone adds a top-level `from app.repositories ...`
(or anything else reaching firebase_admin) back to app/worker/tasks.py.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

# Reaching any of these in the parent is what breaks the forked children.
FORK_UNSAFE = ("grpc", "grpc._cython.cygrpc", "google.cloud.firestore", "firebase_admin")


def _modules_loaded_after(import_line: str) -> set[str]:
    """Import something in a clean interpreter, report which are loaded.

    A subprocess is the point: pytest has almost certainly imported firebase
    already via another test module, so asking the current interpreter would
    always say "loaded" and the test could never fail for the right reason.
    """
    script = textwrap.dedent(f"""
        import sys
        {import_line}
        print("\\n".join(m for m in {FORK_UNSAFE!r} if m in sys.modules))
    """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return {line for line in proc.stdout.split() if line}


def test_importing_tasks_does_not_load_grpc():
    loaded = _modules_loaded_after("import app.worker.tasks")
    assert not loaded, (
        "app.worker.tasks pulled " + ", ".join(sorted(loaded)) + " into the Celery "
        "parent process. grpc is not fork-safe — move the offending import "
        "inside the task/worker_process_init function body. See the note at "
        "the top of app/worker/tasks.py."
    )


def test_importing_celery_app_does_not_load_grpc():
    loaded = _modules_loaded_after("import app.worker.celery_app")
    assert not loaded, (
        "app.worker.celery_app pulled " + ", ".join(sorted(loaded)) + " into the "
        "Celery parent process. See the note at the top of app/worker/tasks.py."
    )


def test_guard_detects_a_real_firebase_import():
    """The guard above is only meaningful if it can actually catch something."""
    assert _modules_loaded_after("import app.firebase"), (
        "app.firebase no longer loads any fork-unsafe module — either the "
        "FORK_UNSAFE list is stale or the detection is broken."
    )
