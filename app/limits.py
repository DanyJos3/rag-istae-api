"""Límite de consultas a /chat, para que nadie agote el cupo diario del modelo.

Dos reglas, en memoria (se reinician si el servicio se reinicia):

- por persona: como máximo ``per_minute`` preguntas por minuto desde la misma IP;
- en total: como máximo ``per_day`` preguntas por día (UTC) entre todos.

Un valor 0 desactiva esa regla (por ejemplo, en local para las evaluaciones).
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class ChatLimiter:
    def __init__(self, per_minute: int, per_day: int, clock=time.time) -> None:
        self.per_minute = per_minute
        self.per_day = per_day
        self.clock = clock
        self.recent: dict[str, deque[float]] = defaultdict(deque)
        self.day = ""
        self.count = 0
        self.lock = threading.Lock()  # /chat corre en varios hilos

    def check(self, client: str) -> str | None:
        """Registra la pregunta si está permitida; si no, devuelve el motivo."""
        with self.lock:
            now = self.clock()
            today = time.strftime("%Y-%m-%d", time.gmtime(now))
            if today != self.day:
                self.day, self.count = today, 0
            if self.per_day and self.count >= self.per_day:
                return "Se alcanzó el límite diario de consultas del sistema. Intente nuevamente mañana."

            window = self.recent[client]
            while window and now - window[0] >= 60:
                window.popleft()
            if self.per_minute and len(window) >= self.per_minute:
                return "Demasiadas preguntas seguidas. Espere un minuto e intente nuevamente."

            window.append(now)
            self.count += 1
            return None
