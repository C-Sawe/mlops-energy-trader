"""Pipeline simulation: replays real market history through the real pipeline
in a separate, disposable process, so an audience can watch the loop work.

A demonstration aid, with no requirement ID of its own (like the activity
timeline). Nothing it does is evidence for Chapter 5: the windows were chosen
because they show something, and the faults in it are injected.

    scenarios.py  the preset windows and what each shows
    worker.py     the child process: sandbox database, one trading day per step
    manager.py    the parent side: starts, commands, reads and kills the child
"""
