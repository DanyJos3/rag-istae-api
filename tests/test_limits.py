from fastapi.testclient import TestClient

from app.limits import ChatLimiter
from app.main import create_app
from tests.test_rag import assistant, rag_settings


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0  # un día cualquiera, a media tarde (UTC)

    def __call__(self) -> float:
        return self.now


def test_per_minute_limit_is_per_ip_and_resets_after_a_minute() -> None:
    clock = Clock()
    limiter = ChatLimiter(per_minute=2, per_day=0, clock=clock)
    assert limiter.check("1.1.1.1") is None
    assert limiter.check("1.1.1.1") is None
    assert "Espere un minuto" in limiter.check("1.1.1.1")
    assert limiter.check("2.2.2.2") is None  # otra persona no se ve afectada
    clock.now += 60
    assert limiter.check("1.1.1.1") is None


def test_daily_limit_is_shared_and_resets_the_next_day() -> None:
    clock = Clock()
    limiter = ChatLimiter(per_minute=0, per_day=2, clock=clock)
    assert limiter.check("1.1.1.1") is None
    assert limiter.check("2.2.2.2") is None
    assert "límite diario" in limiter.check("3.3.3.3")
    clock.now += 24 * 3600
    assert limiter.check("3.3.3.3") is None


def test_zero_disables_the_limits() -> None:
    limiter = ChatLimiter(per_minute=0, per_day=0)
    assert all(limiter.check("1.1.1.1") is None for _ in range(50))


def test_chat_answers_429_with_a_readable_message_when_over_the_limit() -> None:
    settings = rag_settings(chat_limit_per_minute=1, chat_limit_per_day=0)
    with TestClient(create_app(settings, None, {"C": assistant("C")})) as client:
        headers = {"X-Forwarded-For": "1.1.1.1, 10.0.0.1"}
        assert client.post("/chat", json={"question": "Hola"}, headers=headers).status_code == 200
        blocked = client.post("/chat", json={"question": "Hola"}, headers=headers)
        other = client.post("/chat", json={"question": "Hola"}, headers={"X-Forwarded-For": "2.2.2.2"})
    assert blocked.status_code == 429
    assert "Espere un minuto" in blocked.json()["detail"]
    assert other.status_code == 200
