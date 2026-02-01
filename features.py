# features.py – Feature-Extraction für Modus-Erkennung (CRN-Input)
"""
Extrahiert pro Frame einen kompakten Feature-Vektor aus dem Weltzustand
für die Modus-Erkennung: Objekttypen, Farbverteilung, Spielfeld-Dynamik,
Interaktionsmuster. Wird als zusätzlicher Input für CRN oder für
supervised CRN-Training genutzt.
"""
import math
import numpy as np
from entities import MAP_SIZE, FoodPellet, Flag, Checkpoint, Resource, PuzzleElement, BaseBot


def _color_to_channel(hex_color):
    """Hex #RRGGBB -> (r,g,b) normalisiert 0..1."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (0.5, 0.5, 0.5)
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def extract_mode_features(world):
    """
    Ein Vektor pro Frame für Modus-Erkennung.
    Dimension: FEATURE_DIM (fest). Nutzbar als Input für CRN oder als
    Label-Quelle (modus_id) beim supervised CRN-Training.
    """
    FEATURE_DIM = 64
    out = np.zeros(FEATURE_DIM, dtype=np.float32)
    idx = 0

    # --- Objekttyp-Analyse (ca. 10) ---
    n_food = len(world.food)
    n_flags = sum(1 for o in world.objects if isinstance(o, Flag))
    n_checkpoints = sum(1 for o in world.objects if isinstance(o, Checkpoint))
    n_resources = sum(1 for o in world.objects if isinstance(o, Resource))
    n_puzzle = sum(1 for o in world.objects if isinstance(o, PuzzleElement))
    out[idx] = min(1.0, n_food / 200.0); idx += 1
    out[idx] = min(1.0, n_flags / 4.0); idx += 1
    out[idx] = min(1.0, n_checkpoints / 16.0); idx += 1
    out[idx] = min(1.0, n_resources / 50.0); idx += 1
    out[idx] = min(1.0, n_puzzle / 4.0); idx += 1

    # --- Farbpattern Bots (ca. 8): Team-Verteilung ---
    red, blue, green, neutral = 0, 0, 0, 0
    for b in world.bots:
        if b.died:
            continue
        c = b.color.upper()
        if "FF0000" in c or c == "#F00": red += 1
        elif "0000FF" in c or c == "#00F": blue += 1
        elif "00FF00" in c or c == "#0F0": green += 1
        else: neutral += 1
    n_alive = max(1, red + blue + green + neutral)
    out[idx] = red / n_alive; idx += 1
    out[idx] = blue / n_alive; idx += 1
    out[idx] = green / n_alive; idx += 1
    out[idx] = neutral / n_alive; idx += 1

    # --- Spielfeld: dynamische Grenze (Battle Royale), zentrale Zone ---
    zone_radius = getattr(world.current_mode, "zone_radius", None)
    if zone_radius is not None:
        out[idx] = min(1.0, zone_radius / MAP_SIZE); idx += 1
    else:
        out[idx] = 0.0; idx += 1
    hill_r = getattr(world.current_mode, "hill_r", None)
    if hill_r is not None:
        out[idx] = 1.0; idx += 1  # King of Hill
    else:
        out[idx] = 0.0; idx += 1

    # --- Symmetrie / Basen: CTF/Team-Flaggen ---
    has_red_flag = any(isinstance(o, Flag) and getattr(o, "team_id", 0) == 1 for o in world.objects)
    has_blue_flag = any(isinstance(o, Flag) and getattr(o, "team_id", 0) == 2 for o in world.objects)
    out[idx] = 1.0 if has_red_flag else 0.0; idx += 1
    out[idx] = 1.0 if has_blue_flag else 0.0; idx += 1

    # --- Sequenzielle Marker (Racing) ---
    n_seq = sum(1 for o in world.objects if isinstance(o, Checkpoint))
    out[idx] = min(1.0, n_seq / 8.0); idx += 1

    # --- Interaktion: Träger-Flagge, Fänger/Zombie ---
    carrying = sum(1 for b in world.bots if not b.died and getattr(b, "carrying_flag", False))
    it_or_zombie = sum(1 for b in world.bots if not b.died and (b.status.get("is_it") or b.status.get("is_zombie")))
    out[idx] = min(1.0, carrying / 5.0); idx += 1
    out[idx] = min(1.0, it_or_zombie / max(1, n_alive)); idx += 1

    # --- Dynamik: Populationsänderung (Infection: viele gleiche Farbe) ---
    out[idx] = green / n_alive if n_alive else 0.0; idx += 1  # Zombies oft grün
    out[idx] = (red + blue) / n_alive if n_alive else 0.0; idx += 1  # Teams

    # --- Puzzle: aktivierte Elemente ---
    activated = sum(1 for o in world.objects if isinstance(o, PuzzleElement) and getattr(o, "activated", False))
    out[idx] = min(1.0, activated / 4.0); idx += 1

    # Rest mit 0 auffüllen bis FEATURE_DIM
    while idx < FEATURE_DIM:
        out[idx] = 0.0
        idx += 1
    return out


def get_mode_id_from_class(mode_class):
    """Konsistente Modus-ID 0..9 für CRN/MSPN-Reihenfolge."""
    from gamemodes import (
        ClassicArena, TagMode, TeamDeathmatchMode, CaptureTheFlagMode, KingOfTheHillMode,
        BattleRoyaleMode, InfectionMode, ResourceCollectorMode, RacingMode, PuzzleCooperationMode
    )
    MODE_ORDER = [
        ClassicArena, TagMode, TeamDeathmatchMode, CaptureTheFlagMode, KingOfTheHillMode,
        BattleRoyaleMode, InfectionMode, ResourceCollectorMode, RacingMode, PuzzleCooperationMode
    ]
    for i, m in enumerate(MODE_ORDER):
        if mode_class == m:
            return i
    return 0
