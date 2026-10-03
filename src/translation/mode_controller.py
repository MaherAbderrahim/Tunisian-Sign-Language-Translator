"""
Mode Controller - Switches between sign recognition and finger spelling modes.

Two modes:
  - SIGN: standard 57-sign recognition (existing pipeline)
  - SPELL: finger spelling with 31-label alphabet

Mode is toggled by:
  - Special TSL signs (SPELL_ON / SPELL_OFF)
  - Manual UI toggle
"""

from typing import Dict


# Special TSL signs that trigger mode changes
# These are existing signs from the 57-class vocabulary repurposed as mode toggles
SPELL_ON_SIGN = "ta3raf"    # "tu connais" / "you know" → enter spell mode
SPELL_OFF_SIGN = "labes"    # "bien" / "well" → exit spell mode


class ModeController:
    """
    Gère la bascule entre mode signe et mode doigté.

    Le mode peut être changé par:
    - Des signes TSL dédiés (SPELL_ON_SIGN, SPELL_OFF_SIGN)
    - Un contrôle manuel via toggle() ou set_mode()
    """

    def __init__(
        self,
        spell_on_sign: str = SPELL_ON_SIGN,
        spell_off_sign: str = SPELL_OFF_SIGN,
        initial_mode: str = "sign",
    ):
        """
        Args:
            spell_on_sign: Signe TSL qui active le mode doigté
            spell_off_sign: Signe TSL qui désactive le mode doigté
            initial_mode: Mode initial ("sign" ou "spell")
        """
        self.spell_on_sign = spell_on_sign
        self.spell_off_sign = spell_off_sign
        self._mode = initial_mode
        self._previous_mode = "sign"

    @property
    def mode(self) -> str:
        """Mode actuel: 'sign' ou 'spell'."""
        return self._mode

    @property
    def is_spell_mode(self) -> bool:
        return self._mode == "spell"

    @property
    def is_sign_mode(self) -> bool:
        return self._mode == "sign"

    def update(self, tsl_label: str) -> Dict[str, object]:
        """
        Vérifie si le signe détecté est un signe de contrôle de mode.
        Si oui, bascule le mode.

        Args:
            tsl_label: Label du signe TSL détecté

        Returns:
            Dict avec:
            {
                "mode": str,              # Mode après mise à jour
                "mode_changed": bool,     # True si le mode a changé
                "action": str,            # "none" | "enter_spell" | "exit_spell"
                "consumed": bool,         # True si le signe a été consommé (pas de traduction)
            }
        """
        if tsl_label == self.spell_on_sign and not self.is_spell_mode:
            self._previous_mode = self._mode
            self._mode = "spell"
            return {
                "mode": self._mode,
                "mode_changed": True,
                "action": "enter_spell",
                "consumed": True,
            }

        if tsl_label == self.spell_off_sign and self.is_spell_mode:
            self._previous_mode = self._mode
            self._mode = "sign"
            return {
                "mode": self._mode,
                "mode_changed": True,
                "action": "exit_spell",
                "consumed": True,
            }

        return {
            "mode": self._mode,
            "mode_changed": False,
            "action": "none",
            "consumed": False,
        }

    def toggle(self) -> str:
        """Bascule le mode manuellement. Retourne le nouveau mode."""
        if self._mode == "sign":
            self._previous_mode = self._mode
            self._mode = "spell"
        else:
            self._previous_mode = self._mode
            self._mode = "sign"
        return self._mode

    def set_mode(self, mode: str):
        """Définit le mode directement."""
        if mode not in ("sign", "spell"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'sign' or 'spell'.")
        self._previous_mode = self._mode
        self._mode = mode

    def get_state(self) -> Dict[str, str]:
        """Retourne l'état complet du contrôleur."""
        return {
            "mode": self._mode,
            "previous_mode": self._previous_mode,
            "spell_on_sign": self.spell_on_sign,
            "spell_off_sign": self.spell_off_sign,
        }
