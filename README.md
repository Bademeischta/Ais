# Multi-Modus Meta-Learning Arena für Adaptive KI-Agenten

Willkommen in der **Ais Meta-Learning Arena**! Dieses Projekt ist ein hochmodernes System zur Erforschung von Meta-Reinforcement Learning. KI-Agenten lernen hier nicht nur ein einzelnes Spiel, sondern entwickeln die Fähigkeit, Spielregeln dynamisch zu erkennen und ihr Verhalten adaptiv anzupassen.

---

## 🚀 Projektvision

Das Ziel ist ein System, das echte Generalisierungsfähigkeiten demonstriert. Agenten nutzen eine hierarchische Netzwerkarchitektur, um innerhalb weniger Frames zu erkennen: "Aha, das ist ein Fangspiel" oder "Das ist Capture the Flag", und aktivieren sofort die entsprechende Experten-Strategie.

### Kernfeatures:
- **10 Dynamische Spielmodi**: Von Classic Arena (Agar.io-Style) über CTF bis hin zu Puzzle Cooperation.
- **Hierarchische KI-Architektur**: Shared Encoder -> Context Recognition Network (CRN) -> Mode-Specific Policy Networks (MSPN).
- **RTX 5070 Optimierung**: Mixed Precision Training (FP16) und persistente Worker für maximale Trainingsgeschwindigkeit.
- **Smart Environment Detection**: Automatisches Setup für Google Colab (Cloud) oder lokale Hochleistungsrechner.

---

## 🛠 Installation & Setup

### Voraussetzungen
- Python 3.10+
- CUDA-fähige GPU (empfohlen: RTX 30er/40er Serie für lokales Training)
- Node.js (optional für Frontend-Entwicklung)

### Lokale Installation (Windows)
1. **Python 3.10+** installieren und bei der Installation **„Add Python to PATH“** aktivieren.
2. Projektordner öffnen (z. B. `C:\Users\...\Desktop\Ais`).
3. Abhängigkeiten einmalig installieren:
   ```bash
   pip install -r requirements.txt
   ```
4. **Starten:** Doppelklick auf `run.bat` oder in der Konsole:
   ```bash
   python app.py
   ```
5. Im Browser **http://localhost:5000** öffnen.

*(Ohne Repository: Einfach den Ordner `Ais` auf deinem PC behalten und `run.bat` ausführen.)*

---

## 🖥 Betrieb & Modi

### Starten des Systems
Das Projekt ist in 4 modulare Zellen unterteilt (ideal für Colab, aber auch lokal ausführbar):

1. **Cell 1 (Setup)**: Erkennt die Umgebung. Setzt Pfade und Hardware-Parameter.
2. **Cell 2 (Frontend)**: Generiert das HTML5-Canvas Interface.
3. **Cell 3 (Backend)**: Initialisiert die Engine und das Meta-Learning System.
4. **Cell 4 (Execution/Training)**: Startet den Server oder den Trainings-Loop.

**Lokal starten (Windows):** `run.bat` doppelklicken oder `python app.py`. Dann im Browser `http://localhost:5000` öffnen.

### Environment Detection (ENV_CONFIG)
Das System skaliert automatisch:
- **Cloud Mode (Colab)**: Nutzt Google Drive für Checkpoints, kleine Batches (64), 2 Worker.
- **Local Mode (RTX 5070)**: Nutzt lokale Pfade, CUDNN Benchmarks, große Batches (512+), 8+ Worker.

---

## 🧠 Das Meta-Learning System

### Die Architektur
1. **Shared Encoder**: Komprimiert 228 Eingabewerte (Vision Rays + Status) in einen 128-dim latenten Raum.
2. **CRN (Context Recognition)**: Ein LSTM-Netzwerk, das zeitliche Muster analysiert und den Modus vorhersagt.
3. **MSPN (Experts)**: 10 spezialisierte Actor-Critic Köpfe, einer pro Spielmodus.
4. **MCN (Meta-Controller)**: Entscheidet über die Gewichtung der Experten.

### Trainings-Pipeline (4 Phasen)
- **Phase 0: Data Collection**: Sammeln von Beobachtungen mit regelbasierten Bots.
- **Phase 1: CRN Training**: Supervised Learning der Modus-Erkennung (Ziel: >90% Accuracy).
- **Phase 2: Expert Training**: Isolierte Optimierung der MSPNs auf ihre jeweiligen Modi.
- **Phase 3: Meta-Training**: Training des MCN in einer gemischten Umgebung mit schnellen Moduswechseln.

---

## 🎮 Spielmodi Details

| Modus | Ziel | Spezialmechanik |
| :--- | :--- | :--- |
| **Classic Arena** | Masse sammeln | Größenbasiertes Fressen |
| **Tag/Fangen** | Nicht "ES" sein | Rollentausch bei Berührung |
| **Team DM** | Gegner eliminieren | Team-Farbcodierung (Rot/Blau) |
| **Capture the Flag** | Flagge erobern | Flaggen-Dropping bei Tod |
| **King of the Hill** | Zone halten | Kontinuierliche Punkte im Zentrum |
| **Battle Royale** | Überleben | Schrumpfende Todeszone |
| **Infection** | Überlebende infizieren | Exponentielles Wachstum der Zombies |
| **Resource Collector** | Wertvolle Erze sammeln | Gold/Silber/Bronze Ressourcen |
| **Racing** | Checkpoints abfahren | Strenge Sequenz-Logik (1->2->3) |
| **Puzzle Coop** | Schalter aktivieren | Team-Koordination erforderlich |

---

## 📊 Visualisierung & UI
- **Modus-Auswahl**: Dropdown oben rechts – Live-Wechsel des Spielmodus ohne Neustart.
- **Attention Rays**: Meta-Learner zeigen Strahlen (Grün=Futter, Rot=Feind, Gelb=Spezial).
- **CRN-Confidence-Bar**: Grüner Balken links – wie sicher der Meta-Learner den Modus erkennt (≥80% aktiviert MSPN).
- **Active Policy**: Anzeige, welches MSPN gerade aktiv ist (Classic, Tag, CTF, …).
- **Tutorial-Overlay**: Beim Modus-Wechsel 3 Sekunden Kurzerklärung der Regeln.
- **Zonen-Darstellung**: King of the Hill und Battle Royale zeigen die Zone (Kreis/Mitte bzw. schrumpfender Radius).

---

## ⚙️ Konfiguration
- **entities.py**: MAP_SIZE, STARTING_MASS.
- **app.py**: TICK_RATE, FOOD_COUNT, SAVE_DIR.
- **networks.py**: CRN (dropout, return_uncertainty), MSPN, MCN.
- **gamemodes.py**: mode_id, victory_condition, reward_structure, get_render_specs() pro Modus.

---

## 📁 Training & Checkpoints
- **cell4_training.py**: Echtes Training: Phase 0 (Data Collection), Phase 1 (CRN supervised), Phase 2 (MSPN/PPO pro Modus), Phase 3 (MCN Meta-Episoden). Checkpoint-Speicherung, CRN-Evaluation.
- **Checkpoints**: `saves/shared_encoder_*.pth`, `feature_encoder_*.pth`, `crn_model_*.pth`, `mcn_model_*.pth`, `mspn_*_*.pth`.
- **features.py**: Feature-Extraction für Modus-Erkennung (Objekttypen, Farbmuster, Zonen, Interaktionen).

### Vollständiges Training ausführen
```bash
python train_full_pipeline.py
```
Führt Ende-zu-Ende aus: CRN-Datensammlung → CRN supervised Training → MSPN-Training (PPO) pro Modus → MCN Meta-Training → Evaluation (CRN-Genauigkeit, Transfer-Qualität) und speichert Plots/Report unter `logs/`.

### Tests ausführen
```bash
python tests.py
```
Mindestens 16 Unit-Tests für Spielmodi, Netze, Bots und Training.

---

## 📖 Weitere Dokumentation
- **TECH.md**: Architektur, CRN/MSPN/MCN, Feature-Engineering, Reward-Design.
- **USER_GUIDE.md**: Interface, Modus-Wechsel, Statistiken, eigene Modi.
- **ANLEITUNG.md**: Kurzüberblick Einstellungen und Ablauf.

Viel Erfolg beim Training deiner adaptiven Agenten! 🚀
