%%writefile templates/index.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Algorithm Arena</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            background-color: #111;
            color: #fff;
            font-family: 'Courier New', Courier, monospace;
            overflow: hidden;
        }
        #gameCanvas {
            display: block;
        }
        #status {
            position: absolute;
            top: 10px;
            left: 10px;
            padding: 5px 10px;
            background: rgba(0, 0, 0, 0.7);
            border: 1px solid #444;
            font-size: 12px;
            z-index: 100;
        }
    </style>
</head>
<body>
    <div id="status">CONNECTING...</div>
    <canvas id="gameCanvas"></canvas>

    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script>
        const canvas = document.getElementById('gameCanvas');
        const ctx = canvas.getContext('2d');
        const statusEl = document.getElementById('status');

        const MAP_SIZE = 2000;
        let scale = 1;

        function resize() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            scale = Math.min(canvas.width, canvas.height) / MAP_SIZE;
        }

        window.addEventListener('resize', resize);
        resize();

        const socket = io({
            reconnection: true,
            reconnectionAttempts: Infinity,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000
        });

        socket.on('connect', () => {
            console.log('Connected to server');
            statusEl.textContent = 'CONNECTED';
            statusEl.style.color = '#0f0';
        });

        socket.on('disconnect', () => {
            console.log('Disconnected');
            statusEl.textContent = 'RECONNECTING...';
            statusEl.style.color = '#f00';
        });

        socket.on('game_state', (gameState) => {
            render(gameState);
        });

        function render(gameState) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            drawGrid();
            drawBoundaries();

            if (gameState.food) {
                gameState.food.forEach(food => {
                    ctx.beginPath();
                    ctx.arc(food.x * scale, food.y * scale, 5 * scale, 0, Math.PI * 2);
                    ctx.fillStyle = '#00FF00';
                    ctx.fill();
                });
            }

            if (gameState.objects) {
                gameState.objects.forEach(obj => drawObject(obj));
            }

            if (gameState.bots) {
                const sortedBots = [...gameState.bots].sort((a, b) => a.mass - b.mass);
                sortedBots.forEach(bot => drawBot(bot));
            }

            drawHUD(gameState);
        }

        function drawObject(obj) {
            const x = obj.x * scale;
            const y = obj.y * scale;
            const r = obj.radius * scale;

            ctx.beginPath();
            if (obj.type === 'flag') {
                ctx.rect(x - r, y - r, r * 2, r * 2);
            } else if (obj.type === 'checkpoint') {
                ctx.arc(x, y, r, 0, Math.PI * 2);
                ctx.lineWidth = 2;
                ctx.strokeStyle = obj.color;
                ctx.stroke();
                ctx.fillStyle = 'rgba(255,255,0,0.1)';
            } else {
                ctx.arc(x, y, r, 0, Math.PI * 2);
            }
            ctx.fillStyle = obj.color;
            if (obj.activated) ctx.fillStyle = '#fff';
            ctx.fill();
        }

        function drawGrid() {
            ctx.strokeStyle = '#222';
            ctx.lineWidth = 1;
            const gridSize = 100 * scale;

            for (let x = 0; x <= MAP_SIZE * scale; x += gridSize) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, MAP_SIZE * scale);
                ctx.stroke();
            }
            for (let y = 0; y <= MAP_SIZE * scale; y += gridSize) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(MAP_SIZE * scale, y);
                ctx.stroke();
            }
        }

        function drawBoundaries() {
            ctx.strokeStyle = '#f00';
            ctx.lineWidth = 10 * scale;
            ctx.strokeRect(0, 0, MAP_SIZE * scale, MAP_SIZE * scale);
        }

        function drawBot(bot) {
            const x = bot.x * scale;
            const y = bot.y * scale;
            const r = bot.radius * scale;

            // Draw attention rays
            if (bot.rays && bot.algo_name === "Meta-Learner") {
                bot.rays.forEach((ray, i) => {
                    const angle = i * (Math.PI * 2 / 24);
                    const dist = ray[0] * 500 * scale;
                    const opacity = (1.0 - ray[0]) * 0.4;

                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    ctx.lineTo(x + Math.cos(angle) * dist, y + Math.sin(angle) * dist);
                    ctx.strokeStyle = `rgba(255, 255, 255, ${opacity})`;
                    if (ray[2] === 1.0) ctx.strokeStyle = `rgba(0, 255, 0, ${opacity})`; // Food
                    if (ray[3] === 1.0) ctx.strokeStyle = `rgba(255, 0, 0, ${opacity})`; // Enemy
                    if (ray[5] === 1.0) ctx.strokeStyle = `rgba(255, 255, 0, ${opacity})`; // Special
                    ctx.lineWidth = 1;
                    ctx.stroke();
                });
            }

            // Body
            ctx.beginPath();
            ctx.arc(x, y, r, 0, Math.PI * 2);
            ctx.fillStyle = bot.color;
            ctx.fill();
            ctx.strokeStyle = bot.team_id === 1 ? '#f00' : (bot.team_id === 2 ? '#00f' : '#000');
            ctx.lineWidth = 2;
            ctx.stroke();

            // Name tag
            ctx.font = `${Math.max(10, 12 * scale)}px monospace`;
            ctx.fillStyle = '#fff';
            ctx.textAlign = 'center';
            ctx.fillText(`[${bot.algo_name}]`, x, y - r - 15 * scale);

            // Metric
            ctx.font = `${Math.max(8, 10 * scale)}px monospace`;
            ctx.fillStyle = '#aaa';
            ctx.fillText(bot.metric, x, y - r - 3 * scale);

            // Mass
            ctx.fillStyle = '#000';
            ctx.font = `${Math.max(8, 10 * scale)}px monospace`;
            ctx.fillText(Math.round(bot.mass), x, y + 4 * scale);
        }

        function drawHUD(gameState) {
            ctx.font = '20px monospace';
            ctx.fillStyle = '#fff';
            ctx.textAlign = 'left';
            ctx.fillText(`MODE: ${gameState.mode}`, 10, 30);

            ctx.font = '14px monospace';
            ctx.fillText(`Frame: ${gameState.frame}`, 10, 60);
            ctx.fillText(`Alive: ${gameState.bots ? gameState.bots.length : 0}`, 10, 80);

            if (gameState.bots) {
                const leaderboard = [...gameState.bots]
                    .sort((a, b) => b.mass - a.mass)
                    .slice(0, 5);

                ctx.textAlign = 'right';
                ctx.fillText('=== LEADERBOARD ===', canvas.width - 10, 30);
                leaderboard.forEach((bot, i) => {
                    ctx.fillStyle = bot.color;
                    ctx.fillText(`${i+1}. [${bot.algo_name}] ${bot.name}: ${Math.round(bot.mass)}`,
                                 canvas.width - 10, 50 + i * 20);
                });
            }

            drawLegend();
        }

        function drawLegend() {
            const algorithms = [
                {name: 'Random', color: '#888888'},
                {name: 'Rules', color: '#8B4513'},
                {name: 'Field', color: '#00FFFF'},
                {name: 'PID', color: '#FFA500'},
                {name: 'Genetic', color: '#00FF00'},
                {name: 'Q-Table', color: '#FFFF00'},
                {name: 'DQN', color: '#FF0000'},
                {name: 'A2C', color: '#8B00FF'},
                {name: 'Search', color: '#0000FF'},
                {name: 'Ensemble', color: '#FF69B4'},
                {name: 'Novel', color: '#FFD700'}
            ];

            ctx.font = '11px monospace';
            ctx.textAlign = 'left';
            algorithms.forEach((algo, i) => {
                const x = 10 + (i * 85);
                const y = canvas.height - 20;

                ctx.fillStyle = algo.color;
                ctx.fillRect(x, y - 10, 12, 12);
                ctx.fillStyle = '#fff';
                ctx.fillText(algo.name, x + 15, y);
            });
        }
    </script>
</body>
</html>
