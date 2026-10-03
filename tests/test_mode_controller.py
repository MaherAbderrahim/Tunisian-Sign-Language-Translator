"""Tests for ModeController."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.translation.mode_controller import ModeController


class TestModeController:
    def test_initial_mode(self):
        mc = ModeController()
        assert mc.mode == "sign"

    def test_toggle(self):
        mc = ModeController()
        assert mc.toggle() == "spell"
        assert mc.toggle() == "sign"

    def test_auto_switch_on(self):
        mc = ModeController()
        event = mc.update("ta3raf")
        assert event["consumed"] is True
        assert event["mode"] == "spell"

    def test_auto_switch_off(self):
        mc = ModeController()
        mc.update("ta3raf")
        event = mc.update("labes")
        assert event["consumed"] is True
        assert event["mode"] == "sign"

    def test_regular_sign_no_switch(self):
        mc = ModeController()
        event = mc.update("bonjour")
        assert event["consumed"] is False
        assert event["mode"] == "sign"

    def test_set_mode(self):
        mc = ModeController()
        mc.set_mode("spell")
        assert mc.mode == "spell"
        mc.set_mode("sign")
        assert mc.mode == "sign"

    def test_get_state(self):
        mc = ModeController()
        state = mc.get_state()
        assert "mode" in state
        assert "previous_mode" in state
        assert state["mode"] == "sign"
