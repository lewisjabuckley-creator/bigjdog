import pytest

from jarvis.core.intent import IntentKind as K
from jarvis.core.intent import parse
from jarvis.core.personality import clean, failure_report

# Natural phrases from the specification must be understood deterministically (no model needed).
CASES = [
    ("Stop.", K.STOP, None), ("Stop that", K.STOP, "that"), ("cancel the build", K.STOP, "build"),
    ("Continue.", K.RESUME, None), ("Continue the project.", K.RESUME, "project"),
    ("Run that again.", K.RETRY, None), ("Open the other one.", K.OPEN_PROJECT, "other one"),
    ("What happened?", K.DIAGNOSE, None), ("what is wrong?", K.DIAGNOSE, None),
    ("is something wrong with the api?", K.DIAGNOSE, "the api"), ("Where are we?", K.REENTRY, None),
    ("How far along is it?", K.STATUS, "it"), ("Forget that.", K.FORGET, None), ("Forget it.", K.STOP, None),
    ("Use the local model.", K.MODEL_USE, "local"), ("Run the numbers.", K.CHAT, None),
    ("Check why it failed.", K.DIAGNOSE, "it"), ("Log that.", K.LOG_THAT, None),
    ("Keep an eye on it.", K.MONITOR, "it"), ("Tell me when it's done.", K.MONITOR, "it"),
    ("Do the same thing for the other project.", K.REPEAT_FOR, "other"),
    ("What are you doing?", K.STATUS, None), ("Why did you do that?", K.WHY, None),
    ("What did you do?", K.WHAT_DID_YOU_DO, None), ("Where did you get that?", K.PROVENANCE, None),
    ("What time is it?", K.TIME, None), ("Focus mode.", K.MODE, None), ("Quiet.", K.MODE, None),
    ("Remember that the staging server is on port 8443", K.REMEMBER, None),
    ("Don't remember this.", K.DONT_REMEMBER, None),
    ("What do you remember about the robotics project?", K.RECALL, "the robotics project"),
    ("Delete everything you remember about that project.", K.FORGET, "that project"),
    ("Why did we choose PostgreSQL?", K.DECISION_WHY, "PostgreSQL"),
    ("Run the tests, fix whatever is obvious, and let me know if anything serious remains.", K.RUN_TESTS, None),
    ("Actually, don't deploy yet.", K.MODIFY_PLAN, "deploy"), ("Open the robotics project.", K.OPEN_PROJECT, "robotics"),
    ("Morning.", K.BRIEFING, None), ("What changed?", K.WHAT_CHANGED, None), ("What's wrong?", K.DIAGNOSE, None),
    ("Why is the application slow?", K.DIAGNOSE, "the application"), ("How's the build?", K.STATUS, "build"),
    ("Are we good?", K.STATUS, None), ("Anything I need to know?", K.STATUS, None),
    ("Watch this folder ~/Downloads", K.MONITOR, "~/Downloads"), ("Keep an eye on the server.", K.MONITOR, "the server"),
    ("Proceed.", K.APPROVE, None), ("No.", K.DENY, None), ("Short version.", K.SHORTER, None),
    ("Explain everything.", K.LONGER, None), ("What are you?", K.SELF, None),
    ("Prepare this project for deployment.", K.CHAT, None), ("Make this faster.", K.CHAT, None),
    ("unload the vision model", K.MODEL_UNLOAD, "vision"), ("Jarvis, what's running?", K.STATUS, None),
]


@pytest.mark.parametrize("text,kind,target", CASES)
def test_spec_phrases(text, kind, target):
    intent = parse(text)
    assert intent.kind == kind, (text, intent)
    assert intent.target == target, (text, intent.target)


def test_parameters_and_flags():
    assert parse("Tell me when it's done.").params == {"notify": "urgent", "until_done": True}
    assert parse("Focus mode").params == {"mode": "focus", "on": True}
    assert parse("exit focus mode").params == {"mode": "focus", "on": False}
    assert parse("Remember that the API key is in the vault").params["content"] == "the API key is in the vault"
    assert parse("dry run: clean this folder").dry_run
    assert parse("$ ls -la").params["command"] == "ls -la"
    assert parse("run `git status`").params["command"] == "git status"
    assert parse("run the tests, fix whatever is obvious").params["rest"] == "fix whatever is obvious"


def test_personality_strips_filler():
    assert clean("Certainly! The build passed. Let me know if you need anything else.") == "The build passed."
    assert clean("Absolutely, here it is.") == "Here it is."


def test_failure_report_format():
    text = failure_report(what="Deployment", why="the health endpoint returned 503", done=["build", "upload"],
                          not_done=["health check"], next_steps=["roll back to 1.8.1"])
    assert text == ("Deployment failed: the health endpoint returned 503. Completed: build and upload. "
                    "Not completed: health check. Next: roll back to 1.8.1.")
