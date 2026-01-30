%%writefile templates/index.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Algorithm Arena</title>
    <style>
        body { margin: 0; padding: 0; background-color: #111; color: #fff; font-family: 'Courier New', Courier, monospace; overflow: hidden; }
        #gameCanvas { display: block; }
        #status { position: absolute; top: 10px; left: 10px; padding: 5px 10px; background: rgba(0, 0, 0, 0.7); border: 1px solid #444; font-size: 12px; z-index: 100; }
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

        const socket = io({ reconnection: true, reconnectionAttempts: Infinity, reconnectionDelay: 1000 });

        socket.on('connect', () => { statusEl.textContent = 'CONNECTED'; statusEl.style.color = '#0f0'; });
        socket.on('disconnect', () => { statusEl.textContent = 'RECONNECTING...'; statusEl.style.color = '#f00'; });
        socket.on('game_state', (gameState) => { render(gameState); });

        function render(gameState) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            drawGrid();
            drawBoundaries();
            if (gameState.food) {
                gameState.food.forEach(f => {
                    ctx.beginPath(); ctx.arc(f.x * scale, f.y * scale, 5 * scale, 0, Math.PI * 2);
                    ctx.fillStyle = '#00FF00'; ctx.fill();
                });
            }
            if (gameState.bots) {
                [...gameState.bots].sort((a, b) => a.mass - b.mass).forEach(bot => drawBot(bot));
            }
            drawHUD(gameState);
        }

        function drawGrid() {
            ctx.strokeStyle = '#222'; ctx.lineWidth = 1;
            const step = 100 * scale;
            for (let x = 0; x <= MAP_SIZE * scale; x += step) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, MAP_SIZE * scale); ctx.stroke(); }
            for (let y = 0; y <= MAP_SIZE * scale; y += step) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(MAP_SIZE * scale, y); ctx.stroke(); }
        }

        function drawBoundaries() { ctx.strokeStyle = '#f00'; ctx.lineWidth = 10 * scale; ctx.strokeRect(0, 0, MAP_SIZE * scale, MAP_SIZE * scale); }

        function drawBot(bot) {
            const x = bot.x * scale, y = bot.y * scale, r = bot.radius * scale;
            ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fillStyle = bot.color; ctx.fill();
            ctx.strokeStyle = '#000'; ctx.lineWidth = 1; ctx.stroke();
            ctx.font = `${Math.max(10, 12 * scale)}px monospace`; ctx.fillStyle = '#fff'; ctx.textAlign = 'center';
            ctx.fillText(`[${bot.algo_name}]`, x, y - r - 15 * scale);
            ctx.font = `${Math.max(8, 10 * scale)}px monospace`; ctx.fillStyle = '#aaa';
            ctx.fillText(bot.metric, x, y - r - 3 * scale);
            ctx.fillStyle = '#000'; ctx.fillText(Math.round(bot.mass), x, y + 4 * scale);
        }

        function drawHUD(gameState) {
            ctx.font = '14px monospace'; ctx.fillStyle = '#fff'; ctx.textAlign = 'left';
            ctx.fillText(`Frame: ${gameState.frame}`, 10, 50);
            if (gameState.bots) {
                const lb = [...gameState.bots].sort((a, b) => b.mass - a.mass).slice(0, 5);
                ctx.textAlign = 'right'; ctx.fillText('=== LEADERBOARD ===', canvas.width - 10, 30);
                lb.forEach((b, i) => { ctx.fillStyle = b.color; ctx.fillText(`${i+1}. [${b.algo_name}] ${b.name}: ${Math.round(b.mass)}`, canvas.width - 10, 50 + i * 20); });
            }
        }
    </script>
</body>
</html>
