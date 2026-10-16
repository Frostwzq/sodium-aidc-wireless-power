"""P1 verification assertions for the configuration contract."""

from src.config import Config


def test_p1_verification_contract() -> None:
    c = Config.for_scenario("B_half_rack")
    assert c.scenario.p_field_target_w == 12_000
    assert 0 < c.load.duty_cycle < 1
    assert set(c.battery.cycle_life_dod.keys()) == {"10%", "20%", "40%", "80%", "100%"}


if __name__ == "__main__":
    test_p1_verification_contract()
    print("P1 verification: 3/3 assertions passed")
