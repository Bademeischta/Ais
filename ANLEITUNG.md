# Ais Meta-Learning Arena – Was du alles machen und einstellen kannst

Diese Anleitung erklärt das Projekt, alle Einstellmöglichkeiten und was wichtig ist.

---

## 1. Was ist das Projekt?

**Ais Arena** ist eine **KI-Spielarena**: Viele Bots (verschiedene Algorithmen) spielen gleichzeitig auf einer 2D-Karte. Einige Bots sind feste Regeln, andere lernen mit Reinforcement Learning (z.B. DQN, Actor-Critic). Drei **Meta-Learner** nutzen ein gemeinsames Netzwerk (Encoder → CRN → MSPN → MCN), um den aktuellen Spielmodus zu erkennen und ihr Verhalten anzupassen.

Du startest den Server lokal, öffnest im Browser **http://localhost:5000** und siehst die Arena live: Bots, Futter, Modi, Leaderboard.

---

## 2. Was du alles machen kannst

### 2.1 Arena ansehen und beobachten
- **Starten:** `run.bat` oder `python app.py` → Browser: http://localhost:5000
- Du siehst: Karte, Bots (farbig nach Algorithmus), Futter (grün), aktueller Modus, Frame-Zähler, Leaderboard (Top 5 nach Masse), Legende unten
- **Meta-Learner** (3 Bots) zeigen „Attention Rays“: Strahlen zeigen, wohin die KI „schaut“ (Grün = Futter, Rot = Feind)
- Kein manuelles Spielen – du schaust nur zu, die Bots steuern sich selbst

### 2.2 Spielmodus ändern (nur im Code)
Aktuell ist **nur ein Modus** beim Start gesetzt: **Classic Arena** (Agar.io-Style: Masse sammeln, andere fressen).

**Wo:** `app.py`, ganz unten im Block `if __name__ == '__main__':`:
```python
world.set_mode(ClassicArena)   # <-- hier Modus wählen
```

**Verfügbare Modi** (alle in `gamemodes.py` definiert):

| Modus (Klasse)        | Kurzbeschreibung                    |
|-----------------------|-------------------------------------|
| `ClassicArena`        | Masse sammeln, Futter fressen       |
| `TagMode`             | Fangen – einer ist „ES“, Berührung tauscht Rolle |
| `TeamDeathmatchMode`  | Zwei Teams (Rot/Blau), Gegner eliminieren |
| `CaptureTheFlagMode`  | Zwei Flaggen, zur Basis bringen     |
| `KingOfTheHillMode`   | Zone in der Mitte halten            |
| `BattleRoyaleMode`    | Überleben, Zone wird kleiner, kein Respawn |
| `InfectionMode`       | Ein Zombie infiziert andere bei Berührung |
| `ResourceCollectorMode`| Gold/Silber/Bronze einsammeln      |
| `RacingMode`          | Checkpoints in Reihenfolge abfahren |
| `PuzzleCooperationMode`| Beide Schalter gleichzeitig aktivieren |

**Beispiel – Tag statt Classic:**
```python
world.set_mode(TagMode)
```
Dafür muss `TagMode` oben in `app.py` bereits importiert sein (ist er).

### 2.3 Bots ändern (Anzahl und Typ)
**Wo:** `app.py`, im gleichen `if __name__ == '__main__':` Block.

- **Liste der Standard-Bots:**  
  `bt = [RandomBot, RuleBasedBot, PotentialFieldBot, PIDControllerBot, GeneticBot, TabularQLearningBot, DeepQBot, ActorCriticBot, HeuristicSearchBot, EnsembleBot, NovelBot]`
- Du kannst Klassen **entfernen** oder **doppelt** eintragen, um weniger/mehr Bots zu haben
- Danach werden **immer 3 MetaBot** angehängt (nutzen Encoder/CRN/MSPN/MCN)

**Bot-Typen (Kurz):**
- **RandomBot** – zufällige Aktionen  
- **RuleBasedBot** – Flucht vor Feinden, Futter ansteuern  
- **PotentialFieldBot** – Kräfte zu Futter, weg von Feinden  
- **PIDControllerBot** – Regelung zum Ziel  
- **GeneticBot** – evolutionärer Algorithmus  
- **TabularQLearningBot** – Q-Learning mit Tabelle  
- **DeepQBot** – DQN (Neuronales Netz)  
- **ActorCriticBot** – Actor-Critic (A2C-Style)  
- **HeuristicSearchBot** – heuristische Suche  
- **EnsembleBot** – Kombination mehrerer Strategien  
- **NovelBot** – Synergy-Netz, Episoden-Memory  
- **MetaBot** – Meta-Learner (CRN + MSPN + MCN)

### 2.4 Training der Meta-KI (getrennt vom Live-Server)
- **Datei:** `cell4_training.py`
- Dort: **Trainer** mit Encoder, CRN, MSPNs, MCN; Worker laufen Episoden in verschiedenen Modi
- **Starten:** Wenn du nur die Arena im Browser nutzt, startest du **nicht** `cell4_training.py`. Training würde man z.B. in einer eigenen Konsole/Skript ausführen (z.B. `python cell4_training.py`, sofern es einen `if __name__ == '__main__'`-Block gibt)
- **cell1_setup.py** wird beim Training genutzt: erkennt Colab vs. lokal, setzt `BATCH_SIZE`, `NUM_WORKERS`, `SAVE_DIR`, aktiviert CUDNN für GPU

### 2.5 Speicherorte (Saves)
- **Wo:** Ordner `saves/` (lokal: `./saves/`)
- **Was:** Alle lernenden Bots speichern periodisch ihre Gewichte (z.B. alle 1000 Frames in `app.py`): `Bot-8.pth`, `Bot-9.pth`, … sowie `crn_model.pth`, `mcn_model.pth`, `shared_encoder_*.pth` usw., wenn das Training sie dort ablegt
- **Wichtig:** Beim nächsten Start **lädt** die App diese Checkpoints aktuell **nicht** automatisch (im Code steht nur „Load weights if available...“). Du könntest dort später Laden-Logik ergänzen

---

## 3. Wichtige Einstellungen (Konfiguration)

### 3.1 Arena & Physik – `entities.py`
| Variable        | Standard | Bedeutung |
|-----------------|----------|-----------|
| `MAP_SIZE`      | 2000     | Kantenlänge der Karte (Pixel/Einheiten) |
| `STARTING_MASS` | 25       | Startmasse jedes Bots (Radius hängt an Masse) |

### 3.2 Server & Spiel – `app.py`
| Variable     | Standard | Bedeutung |
|--------------|----------|-----------|
| `FOOD_COUNT` | 150      | Anzahl Futter-Pellets (bei Classic; bei `spawn_food` neu erzeugt) |
| `BOT_COUNT`  | 22       | Wird nicht direkt genutzt – die echte Anzahl kommt aus der Bot-Liste + 3 MetaBots |
| `TICK_RATE`  | 30       | Ziel-FPS der Spielschleife (30 Updates pro Sekunde) |
| `MAX_BOT_MASS` | 500   | Maximale Masse eines Bots |
| `MIN_BOT_MASS` | 15    | Minimale Masse (darunter „stirbt“ bzw. Respawn) |
| `SAVE_DIR`   | `./saves/` | Ordner für Bot-Checkpoints (.pth) |

### 3.3 Netzwerke – `networks.py`
- **SharedEncoder:** `input_dim=228`, `latent_dim=128` (Eingabe = z.B. Vision Rays + Status)
- **CRN:** LSTM, `latent_dim=128`, `hidden_dim=64`, `num_modes=10`
- **MSPN:** 10 Stück, je Actor-Critic, `latent_dim=128`, `action_dim=12`
- **MCN:** Eingabe Modus-Wahrscheinlichkeiten + Meta-Stats, Ausgabe Gewichtung der Modi
- **DQNet / ACNet / DuelingDQNet:** Für DeepQBot, ActorCriticBot (228 Eingaben, 12 Aktionen)

Wenn du Eingabedimension oder Anzahl Aktionen änderst, musst du das an allen Stellen anpassen, die diese Netze nutzen.

### 3.4 Training (cell4_training.py & cell1_setup)
- **BATCH_SIZE:** 64 (Colab) / 512 (lokal)
- **NUM_WORKERS:** 2 (Colab) / 8 (lokal)
- **SAVE_DIR / LOG_DIR:** über `cell1_setup.ENV_CONFIG`

### 3.5 Frontend – `templates/index.html`
- **MAP_SIZE** im JavaScript muss mit `entities.MAP_SIZE` (2000) übereinstimmen, sonst passt die Skalierung nicht
- Du kannst Texte, Farben, HUD, Legende anpassen; die Spiellogik kommt weiterhin vom Server

---

## 4. Was ist wichtig – Kurzüberblick

1. **Starten:** `run.bat` oder `python app.py` → http://localhost:5000  
2. **Modus:** In `app.py` mit `world.set_mode(ModusKlasse)` setzen; nur beim Start, kein Wechsel per Klick.  
3. **Bots:** In `app.py` die Liste `bt` anpassen; 3 MetaBots kommen immer dazu.  
4. **Weltgröße / Masse:** `entities.py` – `MAP_SIZE`, `STARTING_MASS`.  
5. **Futter, Tick-Rate, Saves:** `app.py` – `FOOD_COUNT`, `TICK_RATE`, `SAVE_DIR`.  
6. **Speicherung:** Bots schreiben in `saves/`; automatisches Laden beim Start ist derzeit nicht implementiert.  
7. **Training:** Eigenes Skript `cell4_training.py` (+ `cell1_setup.py` für Umgebung); läuft getrennt von der Live-Arena.  
8. **Netzwerk-Dimensionen:** In `networks.py` – nur ändern, wenn du die Beobachtungs-/Aktionsräume anpasst.

Wenn du eine konkrete Änderung (z.B. „nur 5 Bots“ oder „Tag-Modus als Standard“) umsetzen willst, kannst du die genannten Stellen direkt im Code anpassen.
