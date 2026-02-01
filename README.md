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

### Lokale Installation
1. Repository klonen:
   ```bash
   git clone https://github.com/your-repo/ais-meta-learning.git
   cd ais-meta-learning
   ```
2. Abhängigkeiten installieren:
   ```bash
   pip install flask flask-socketio eventlet numpy torch psutil pyngrok
   ```

---

## 🖥 Betrieb & Modi

### Starten des Systems
Das Projekt ist in 4 modulare Zellen unterteilt (ideal für Colab, aber auch lokal ausführbar):

1. **Cell 1 (Setup)**: Erkennt die Umgebung. Setzt Pfade und Hardware-Parameter.
2. **Cell 2 (Frontend)**: Generiert das HTML5-Canvas Interface.
3. **Cell 3 (Backend)**: Initialisiert die Engine und das Meta-Learning System.
4. **Cell 4 (Execution/Training)**: Startet den Server oder den Trainings-Loop.

**Lokal starten:**
```bash
python3 app.py
```
Öffne dann `http://localhost:5000` im Browser.

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

## 📊 Visualisierung & Debugging
- **Attention Rays**: Der Meta-Learner zeigt seine "Aufmerksamkeit" durch farbige Strahlen (Grün=Food, Rot=Feind).
- **Mode Indicator**: Live-Anzeige der erkannten Spielregeln oben links.
- **Confidence Bar**: Zeigt an, wie sicher sich die KI über den aktuellen Modus ist.

---

## ⚙️ Konfiguration
Alle wichtigen Parameter befinden sich in `entities.py` (MAP_SIZE, STARTING_MASS) und `app.py` (TICK_RATE, FOOD_COUNT).
Die Netzwerk-Hyperparameter können in `networks.py` angepasst werden.

Viel Erfolg beim Training deiner adaptiven Agenten! 🚀
