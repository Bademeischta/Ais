# User Guide – Ais Meta-Learning Arena

## Interface nutzen

1. **Starten**: `run.bat` oder `python app.py` → Browser **http://localhost:5000**.
2. **Verbindung**: Oben links „CONNECTED“ (grün) = Server verbunden.
3. **Automatischer Moduswechsel (AFK-Training)**: Ist standardmäßig **an**. Die Arena wechselt alle **~100 Sekunden** (3000 Frames) automatisch zum nächsten Modus (Classic → Tag → TDM → CTF → … → Puzzle → Classic). So kannst du die Arena laufen lassen und die KIs trainieren in allen Modi. Links im HUD steht z. B. „Auto-Rotation: an | Nächster: Tag in 45s“. Ausschalten/Intervall ändern: in `app.py` `AUTO_MODE_ROTATION = False` oder `MODE_SWITCH_INTERVAL_FRAMES = 6000` (z. B. 200 s).
4. **Modus manuell wechseln**: Oben rechts im **Dropdown** einen der 10 Modi wählen. Der Modus wechselt sofort; ein kurzes **Tutorial-Overlay** erklärt die Regeln.
4. **Anzeigen**:
   - **MODE**: aktueller Spielmodus.
   - **Grüner Balken** (links): Modus-Confidence des Meta-Learners (0–100 %). Ab ~80 % nutzt die KI die erkannte Experten-Policy.
   - **Active Policy**: rechts unter dem Leaderboard – welcher Experte (Classic, Tag, CTF, …) gerade aktiv ist.
   - **Leaderboard**: Top 5 nach Masse; Farben = Bot-Typ (Legende unten).
5. **Zonen**: Bei **King of the Hill** und **Battle Royale** wird die Zone (gelber Kreis bzw. schrumpfender Radius) gezeichnet.

## Statistiken interpretieren

- **Frame**: Laufende Spielzeit in Frames (30 Frames ≈ 1 Sekunde).
- **Alive**: Anzahl lebender Bots.
- **Metric** unter jedem Bot: z. B. „Classic (85 %)“ = Meta-Learner erkennt Classic Arena mit 85 % Sicherheit; „Scanning…“ = noch unsicher.
- **Attention Rays** (nur Meta-Learner): Strahlen zeigen, wohin die KI „schaut“ (Grün = Futter, Rot = Feind, Gelb = Spezialobjekt).

## Modus-Wechsel

- Während des Laufs im **Dropdown** einen anderen Modus wählen.
- **Alle Bots respawnen automatisch** im neuen Modus (leben wieder, neue Position, Startmasse) – auch nach Battle Royale, wo im Modus selbst kein Respawn ist.
- Die **gelernten Gewichte** (DQN, A2C, Genetic, TabularQ, Novel, Meta-Learner) **bleiben erhalten**. Die KIs sammeln im neuen Modus weiter Erfahrung und lernen so mit der Zeit jeden Modus – du lässt es laufen, wechselst ab und zu den Modus, und irgendwann können sie jeden Modus spielen.
- Kein Neustart nötig.

## Eigene Modi hinzufügen

1. In **gamemodes.py** eine neue Klasse anlegen, die von `GameMode` erbt.
2. `initialize()`, `update()`, `calculate_reward()`, `check_victory()` implementieren; optional `get_render_specs()`.
3. `mode_id` und `name` setzen; in `MODE_IDS` und `MODES` eintragen.
4. In **app.py** die neue Modus-Klasse importieren und in `MODES` aufnehmen.
5. Im **Frontend** (`templates/index.html`) eine neue Option im `<select id="modeSelect">` und einen Eintrag in `TUTORIALS` hinzufügen.
6. In **networks.py** ggf. `num_modes` erhöhen und ein weiteres MSPN hinzufügen (CRN/MCN dann 11 Modi).

## Training

- **Nur Arena ansehen**: `python app.py` reicht.
- **Training (Curriculum + Checkpoints)**: `python cell4_training.py`. Führt Phase 1 (isoliert), Phase 2 (gemischt), Phase 3 (Rapid Switch) und eine kurze CRN-Evaluation aus. Checkpoints landen in `saves/`.
- **Resume**: In `cell4_training.py` z. B. `trainer.load_checkpoint("phase2")` vor weiteren Läufen aufrufen.

## FAQ

- **Seite bleibt weiß**: Prüfen, ob Server läuft (Konsole: „Ais Arena läuft lokal“) und ob **http://localhost:5000** (nicht https) verwendet wird.
- **Modus wechselt nicht**: Browser-Cache leeren oder Hard-Reload (Strg+F5); prüfen, ob Socket.IO verbunden ist („CONNECTED“).
- **Meta-Learner zeigt immer „Scanning…“**: CRN braucht mind. 50 Frames Beobachtung; Confidence unter 80 % → Anzeige „Scanning…“ oder niedrige Prozent. Nach Modus-Wechsel oder mit wenig erkennbaren Merkmalen kann die Confidence länger niedrig bleiben.
- **Checkpoints laden**: Aktuell lädt die Live-Arena (`app.py`) beim Start keine Gewichte. Training nutzt `load_checkpoint()` in `cell4_training.py`. Laden in `app.py` kann ergänzt werden („Load weights if available“).
