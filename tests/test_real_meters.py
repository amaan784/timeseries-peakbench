from src.real_meters import battery_for, building_ids


def test_battery_is_sized_from_the_peak():
    bat = battery_for(232.7)  # 30% of the peak, rounded to 5 kW, two hours
    assert bat.power_kw == 70
    assert bat.capacity_kwh == 140


def test_the_six_picked_buildings_are_on_disk():
    from src.uci import OUT_DIR

    ids = building_ids()
    assert len(ids) == 6
    for b in ids:
        assert (OUT_DIR / f"{b}.csv").exists()
