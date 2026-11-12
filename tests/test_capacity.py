from proctor_assist import capacity


def test_search_finds_the_largest_real_time_number(monkeypatch):
    tried = []

    def fake(model, n, places, subjects, seconds=capacity.SECONDS):
        tried.append(n)
        return {"max_wall_s": n, "real_time": n <= 37}

    monkeypatch.setattr(capacity, "trial", fake)
    rows, best = capacity.search("p1", [], [1])
    assert best == 37
    assert tried[:7] == [1, 2, 4, 8, 16, 32, 64]
    assert len(rows) == len(tried)


def test_search_stops_at_the_cap_and_at_zero(monkeypatch):
    monkeypatch.setattr(capacity, "trial", lambda m, n, p, s, seconds=60: {"max_wall_s": 1, "real_time": True})
    assert capacity.search("r0", [], [1])[1] == capacity.MAX_STREAMS
    monkeypatch.setattr(capacity, "trial", lambda m, n, p, s, seconds=60: {"max_wall_s": 99, "real_time": False})
    assert capacity.search("e2", [], [1])[1] == 0
