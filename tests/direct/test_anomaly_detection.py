"""Tests for the AnomalyDetection vector-semantic contract."""


def test_default_threshold(direct_deploy):
    default = direct_deploy("contracts/anomaly_detection.py")
    assert default.novel_threshold == 0.80


def test_custom_constructor_threshold(direct_deploy):
    strict = direct_deploy("contracts/anomaly_detection.py", novel_threshold=95)
    assert strict.novel_threshold == 0.95


def test_empty_store_first_observation_is_novel(direct_deploy):
    contract = direct_deploy("contracts/anomaly_detection.py")
    assert contract.observation_count() == 0
    assert contract.is_novel("network latency spike on node-7") is True
    assert contract.get_closest("anything") is None


def test_similar_observation_is_not_novel(direct_vm, direct_deploy, direct_alice):
    contract = direct_deploy("contracts/anomaly_detection.py")
    direct_vm.sender = direct_alice

    result = contract.add_observation("disk usage above 90% on server-a")
    assert result["is_novel"] is True
    assert result["reason"] == "No prior observations to compare against."

    direct_vm.mock_llm(
        ".*disk usage above 90% on server-a.*",
        "DUPLICATE The same disk-usage incident on server-a is already recorded.",
    )
    second = contract.add_observation("disk usage above 90% on server-a")
    assert second["is_novel"] is False
    assert second["reason"].startswith("DUPLICATE")

    assert contract.observation_count() == 2


def test_distinct_observation_is_novel(direct_vm, direct_deploy, direct_alice):
    contract = direct_deploy("contracts/anomaly_detection.py")
    direct_vm.sender = direct_alice

    contract.add_observation("database connection pool exhausted")
    direct_vm.mock_llm(
        ".*cpu temperature nominal.*",
        "NOVEL A nominal CPU temperature is a new, unrelated observation.",
    )
    second = contract.add_observation("cpu temperature nominal")
    assert second["is_novel"] is True
    assert second["reason"].startswith("NOVEL")


def test_malformed_verdict_falls_back_to_threshold(
    direct_vm, direct_deploy, direct_alice
):
    contract = direct_deploy("contracts/anomaly_detection.py")
    direct_vm.sender = direct_alice

    contract.add_observation("zookeeper leader election failed")
    direct_vm.mock_llm(".*zookeeper leader election failed.*", "maybe")
    duplicate = contract.add_observation("zookeeper leader election failed")
    assert duplicate["is_novel"] is False


def test_validator_accepts_and_rejects_verdicts(
    direct_vm, direct_deploy, direct_alice
):
    contract = direct_deploy("contracts/anomaly_detection.py")
    direct_vm.sender = direct_alice

    contract.add_observation("payment gateway timeout")

    direct_vm.mock_llm(
        ".*payment gateway timeout.*",
        "NOVEL A genuinely new payment incident.",
    )
    contract.add_observation("payment gateway timeout")
    assert direct_vm.run_validator() is True

    direct_vm.clear_mocks()
    direct_vm.mock_llm(
        ".*payment gateway timeout.*",
        "unclear response",
    )
    contract.add_observation("payment gateway timeout")
    assert direct_vm.run_validator() is False


def test_get_closest_returns_best_match(direct_vm, direct_deploy, direct_alice):
    contract = direct_deploy("contracts/anomaly_detection.py")
    direct_vm.sender = direct_alice

    contract.add_observation("payment gateway timeout")
    direct_vm.mock_llm(
        ".*payment gateway timeout.*",
        "DUPLICATE The same payment gateway timeout is already recorded.",
    )
    contract.add_observation("payment gateway timeout")
    direct_vm.mock_llm(
        ".*customer refund processed.*",
        "NOVEL A customer refund is a different kind of event.",
    )
    contract.add_observation("customer refund processed")

    closest = contract.get_closest("payment gateway timeout")
    assert closest is not None
    assert closest["text"] == "payment gateway timeout"
    assert isinstance(closest["similarity"], str)


def test_remove_observation(direct_vm, direct_deploy, direct_alice):
    contract = direct_deploy("contracts/anomaly_detection.py")
    direct_vm.sender = direct_alice

    result = contract.add_observation("zookeeper leader election failed")
    log_id = int(result["log_id"])
    assert contract.observation_count() == 1

    removal = contract.remove_observation(log_id)
    assert removal["removed"] is True
    assert contract.observation_count() == 0