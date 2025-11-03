"""
DIFFUSION SWITCHING: Dynamische Änderung der Diffusionsart
===========================================================

PHYSIKALISCHE MOTIVATION:
-------------------------
In realen Hydrogelen können Partikel zwischen verschiedenen Diffusionsarten
wechseln:

1. NORMAL → CONFINED: Partikel wird in Pore gefangen
2. CONFINED → NORMAL: Partikel entkommt aus Pore
3. NORMAL → SUB: Partikel trifft auf Netzwerk-Hindernis
4. SUB → NORMAL: Partikel überwindet Hindernis

Switching-Wahrscheinlichkeit hängt ab von:
- Polymerisationszeit (mehr Vernetzung → häufigeres Switching)
- Aktuellem Diffusionstyp
- Lokaler Netzwerkdichte

IMPLEMENTIERUNG:
----------------
Jedes Partikel hat eine Switching-Wahrscheinlichkeit pro Frame.
Bei Switch wird neuer Diffusionstyp basierend auf aktuellen Fraktionen gewählt.

Referenzen:
-----------
- Metzler et al. (2014): Anomalous diffusion models
- Weigel et al. (2011): Ergodic and nonergodic processes
"""

from typing import Dict, Tuple
import random


class DiffusionSwitcher:
    """
    Verwaltet dynamisches Switching zwischen Diffusionsarten.
    """

    def __init__(self, t_poly_min: float, base_switch_prob: float = 0.01):
        """
        Parameters:
        -----------
        t_poly_min : float
            Polymerisationszeit [min]
        base_switch_prob : float
            Basis-Switching-Wahrscheinlichkeit pro Frame (default: 1%)
        """
        self.t_poly_min = t_poly_min
        self.base_switch_prob = base_switch_prob

        # Berechne Switching-Wahrscheinlichkeit basierend auf Vernetzungsgrad
        self.switch_prob = self._calculate_switch_probability()

    def _calculate_switch_probability(self) -> float:
        """
        Berechnet Switching-Wahrscheinlichkeit basierend auf Polymerisationszeit.

        Logik:
        ------
        - t < 30 min: Wenig Switching (homogenes Medium)
        - t = 30-90 min: Zunehmendes Switching (Netzwerk bildet sich)
        - t > 90 min: Hohes Switching (heterogenes Netzwerk)
        """

        if self.t_poly_min < 30:
            # Früh: wenig Switching
            return self.base_switch_prob * 0.5

        elif self.t_poly_min < 90:
            # Mittlere Phase: linearer Anstieg
            progress = (self.t_poly_min - 30.0) / 60.0
            return self.base_switch_prob * (0.5 + 2.0 * progress)  # 0.5x → 2.5x

        else:
            # Spät: viel Switching (heterogenes Netzwerk)
            return self.base_switch_prob * 2.5

    def should_switch(self, current_type: str) -> bool:
        """
        Entscheidet, ob ein Switch stattfinden soll.

        Parameters:
        -----------
        current_type : str
            Aktueller Diffusionstyp

        Returns:
        --------
        bool : True wenn Switch erfolgen soll
        """

        # Typ-spezifische Modifikation
        type_modifier = {
            "normal": 1.0,       # Normal: Standard-Switching
            "subdiffusion": 0.7,  # Sub: etwas stabiler (im Netzwerk gefangen)
            "confined": 1.5,      # Confined: instabil (versucht zu entkommen)
            "superdiffusion": 2.0  # Super: sehr instabil (Konvektion stoppt)
        }

        effective_prob = self.switch_prob * type_modifier.get(current_type, 1.0)

        return random.random() < effective_prob

    def get_new_type(self, current_type: str,
                     fractions: Dict[str, float]) -> str:
        """
        Wählt neuen Diffusionstyp basierend auf physikalischen Übergängen.

        Parameters:
        -----------
        current_type : str
            Aktueller Typ
        fractions : Dict[str, float]
            Aktuelle Fraktionen aller Typen

        Returns:
        --------
        str : Neuer Diffusionstyp
        """

        # Definiere erlaubte Übergänge (physikalisch sinnvoll)
        transitions = {
            "normal": {
                "normal": 0.2,        # Bleibt normal
                "subdiffusion": 0.5,  # Trifft auf Hindernis
                "confined": 0.3,      # Wird gefangen
                "superdiffusion": 0.0  # Kein Übergang zu super
            },
            "subdiffusion": {
                "normal": 0.6,        # Überwindet Hindernis
                "subdiffusion": 0.2,  # Bleibt sub
                "confined": 0.2,      # Wird stärker gefangen
                "superdiffusion": 0.0
            },
            "confined": {
                "normal": 0.5,        # Entkommt!
                "subdiffusion": 0.3,  # Teilweise frei
                "confined": 0.2,      # Bleibt gefangen
                "superdiffusion": 0.0
            },
            "superdiffusion": {
                "normal": 0.8,        # Konvektion stoppt → normal
                "subdiffusion": 0.2,  # Direkt ins Netzwerk
                "confined": 0.0,
                "superdiffusion": 0.0  # Super verschwindet
            }
        }

        # Hole Übergangswahrscheinlichkeiten
        trans_probs = transitions.get(current_type, {})

        # Wähle neuen Typ
        types = list(trans_probs.keys())
        probs = list(trans_probs.values())

        # Normalisierung
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            # Fallback: verwende globale Fraktionen
            types = list(fractions.keys())
            probs = list(fractions.values())

        # Random Choice
        new_type = random.choices(types, weights=probs)[0]

        return new_type


# ============================================================================
# BEISPIEL-NUTZUNG
# ============================================================================

def simulate_switching_example():
    """Beispiel: Zeigt Switching über Zeit."""

    print("=" * 80)
    print("DIFFUSION SWITCHING - BEISPIEL")
    print("=" * 80)

    # Simuliere für verschiedene Polyzeiten
    for t_poly in [0, 30, 60, 90]:
        print(f"\n📊 Polymerisationszeit: {t_poly} min")
        print("-" * 60)

        switcher = DiffusionSwitcher(t_poly, base_switch_prob=0.05)
        print(f"  Switch-Wahrscheinlichkeit: {switcher.switch_prob*100:.1f}% pro Frame")

        # Simuliere 100 Frames
        current_type = "normal"
        switch_events = []

        for frame in range(100):
            if switcher.should_switch(current_type):
                # Simple fractions für Demo
                fractions = {
                    "normal": 0.6,
                    "subdiffusion": 0.3,
                    "confined": 0.1,
                    "superdiffusion": 0.0
                }
                new_type = switcher.get_new_type(current_type, fractions)

                if new_type != current_type:
                    switch_events.append((frame, current_type, new_type))
                    current_type = new_type

        print(f"  Anzahl Switches: {len(switch_events)}")
        if switch_events:
            print(f"  Beispiele:")
            for frame, old, new in switch_events[:3]:
                print(f"    Frame {frame:3d}: {old:15s} → {new}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    simulate_switching_example()
