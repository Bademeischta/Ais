# Technische Dokumentation – Ais Meta-Learning Arena

## Architektur-Übersicht

```
Raw Observations (228-dim: 24 Rays × 9 + Status)
        ↓
  Shared Encoder (228 → 128)
        ↓
  Latent Sequence (100 Frames) → CRN (LSTM + FC) → Modus-Wahrscheinlichkeiten (10) + Uncertainty
        ↓
  MCN (mode_probs + meta_stats → MSPN-Gewichtung)
        ↓
  MSPN[active_mode] (128 → Actor 12, Critic 1) → Aktion
```

- **Shared Encoder** (`networks.py`): Linear 228→256→128, ReLU. Gemeinsame Repräsentation für alle Modi.
- **CRN**: LSTM(128→64), Dropout, FC→10, Softmax. Optional `return_uncertainty=True` liefert Entropy-basierte Unsicherheit.
- **MSPN**: 10 × (Actor-Critic auf 128-dim Latent). Jeder Modus 0..9 hat ein MSPN.
- **MCN**: Konkateniert Modus-Probs + Meta-Stats (5), FC→32→10, Softmax (Experten-Gewichtung).

## Feature-Extraction für Modus-Erkennung (`features.py`)

- **extract_mode_features(world)** liefert einen 64-dim Vektor pro Frame:
  - Objekttypen: Anteile Food, Flaggen, Checkpoints, Ressourcen, Puzzle.
  - Farbpattern: Anteile Rot/Blau/Grün/Neutral der Bots (Teams vs. Tag/Infection).
  - Spielfeld: zone_radius (Battle Royale), hill_r (King of Hill), Symmetrie/Flaggen.
  - Interaktion: carrying_flag, Fänger/Zombie-Anteil, aktivierte Puzzle-Elemente.
- Nutzbar für supervised CRN-Training (Labels = mode_id) oder als zusätzlicher Input.

## Reward-Design

- **Classic**: Massengewinn, Kills (Größenvorteil), Futter; Strafen für Wand.
- **Tag**: Fänger +100 bei Tag, -50 Opfer; kontinuierlich -0.1 (Fänger) / +0.1 (Läufer).
- **CTF**: +500 Capture, +100 Aufnahme, +0.5 Tragen; Shaped: Annäherung an gegnerische Flagge.
- **King of Hill**: +1.0 in Zone, -0.1 außerhalb.
- **Battle Royale**: +0.2 Überleben, -1 außerhalb Zone, Massenverlust.
- **Infection**: Zombie +100 pro Konversion, Survivor -50; kontinuierlich 0.1 / 0.2.
- **Racing**: +200 pro Checkpoint in Reihenfolge; Shaped: Abstand zum nächsten CP.
- **Puzzle Coop**: +10 für alle, wenn beide Platten aktiv.

## Vision / Raycast (`entities.py`)

- Pro Ray 9 Werte: `[dist, wall, food, friend_foe, is_bot, special, value, size, rel_mass]`.
- `special=1` für Flag, Checkpoint, Resource, Puzzle; `value` modusabhängig (team_id, sequence, resource value, activated).
- Erweiterung für Multi-Channel/Embedding optional (Object-Embedding-System laut Spec).

## Training-Pipeline (`cell4_training.py`)

- **Phase 1 (Isolated)**: Pro Modus N Episoden mit RuleBasedBot; Checkpoints nach Modus-Blöcken.
- **Phase 2 (Mixed)**: Zufälliger Modus pro Episode, Multi-Worker, train_step mit Replay-Daten.
- **Phase 3 (Rapid Switch)**: Modus wechselt alle 200 Frames; Meta-Episoden.
- **Checkpoints**: `shared_encoder_*.pth`, `crn_model_*.pth`, `mcn_model_*.pth`, `mspn_{i}_*.pth`.
- **Evaluation**: `evaluate_crn_accuracy()` (Stub: gelabelte Episoden, CRN-Vorhersage vs. mode_id).

## MetaBot-Logik (`bots.py`)

- Puffer: 100 Frames Beobachtung → Encoder → Latent-Sequenz.
- CRN alle 20 Frames ab Pufferlänge ≥ 50; Ausgabe: mode_probs, uncertainty.
- Confidence = max(mode_probs). Nur wenn **confidence ≥ 0.8** wird das zugehörige MSPN genutzt; sonst Exploration (30 % Zufall, sonst aktuelles MSPN).
- Active Policy = argmax(mode_probs); wird im Frontend angezeigt.

## Hyperparameter (Referenz)

| Komponente | Parameter | typ. Wert |
|------------|-----------|-----------|
| SharedEncoder | input_dim, latent_dim | 228, 128 |
| CRN | hidden_dim, num_modes, dropout | 64, 10, 0.1 |
| MSPN | latent_dim, action_dim | 128, 12 |
| MCN | num_modes, meta_stats_dim | 10, 5 |
| MetaBot | CONFIDENCE_THRESHOLD, MIN_BUFFER_LEN | 0.8, 50 |

## Muss-Kriterien (Spec)

- Alle 10 Modi spielbar: ✅ (gamemodes.py).
- CRN mit Uncertainty: ✅ (return_uncertainty, Entropy).
- Adaption: Confidence-Threshold + Exploration; Ziel >50 % spezialisierte Performance in 100 Episoden (Evaluation ausbaubar).
- Echtzeit ≥20 fps: ✅ (TICK_RATE 30, eventlet).
- Stabilität bei Modus-Wechsel: ✅ (set_mode unter Lock, Frontend Dropdown + SocketIO).
