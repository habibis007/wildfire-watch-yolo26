from wildfire import Detection, PersistenceAlert


def det(cls="smoke", conf=0.8):
    return Detection(cls, conf, 0, 0, 10, 10)


def feed(alert, pattern, cls="smoke", dt=0.5):
    """pattern: string of '1' (detected) / '0' (not). Returns list of raise-times."""
    raised = []
    for i, ch in enumerate(pattern):
        evs = alert.update(i * dt, [det(cls)] if ch == "1" else [])
        raised += [e.raised_time for e in evs]
    return raised


def test_no_alert_below_threshold():
    a = PersistenceAlert(min_consecutive=5)
    assert feed(a, "11110111101111") == []
    assert a.events == []


def test_alert_fires_on_fifth_consecutive_frame():
    a = PersistenceAlert(min_consecutive=5)
    assert feed(a, "0011111") == [3.0]
    ev = a.events[0]
    assert ev.start_time == 1.0 and ev.raised_time == 3.0 and ev.end_time == 3.0


def test_single_alert_for_sustained_event_with_flicker():
    a = PersistenceAlert(min_consecutive=3, clear_after=3)
    # one-frame dropout inside an active event must not create a second event
    assert feed(a, "1111101111") == [1.0]
    assert len(a.events) == 1
    assert a.events[0].end_time == 4.5


def test_new_event_after_clear():
    a = PersistenceAlert(min_consecutive=2, clear_after=2)
    assert feed(a, "11001100") == [0.5, 2.5]
    assert len(a.events) == 2


def test_low_confidence_ignored():
    a = PersistenceAlert(min_consecutive=2, min_confidence=0.5)
    for i in range(5):
        a.update(i, [det(conf=0.3)])
    assert a.events == []


def test_classes_tracked_independently():
    a = PersistenceAlert(min_consecutive=2)
    a.update(0, [det("fire")])
    a.update(1, [det("smoke")])  # breaks fire streak, starts smoke
    assert a.events == []
    a.update(2, [det("smoke"), det("fire")])
    assert [e.class_name for e in a.events] == ["smoke"]
    a.update(3, [det("fire")])
    assert sorted(a.active_classes) == ["fire", "smoke"]
    assert "FIRE" in a.status_text() and "SMOKE" in a.status_text()


def test_unknown_class_ignored():
    a = PersistenceAlert(min_consecutive=1)
    a.update(0, [det("cloud")])
    assert a.events == []
