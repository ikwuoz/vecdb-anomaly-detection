"""Tests for the AnomalyDetection vector-semantic contract."""


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

    second = contract.add_observation("disk usage above 90% on server-a")
    assert second["is_novel"] is False

    assert contract.observation_count() == 2


def test_distinct_observation_is_novel(direct_vm, direct_deploy, direct_alice):
    contract = direct_deploy("contracts/anomaly_detection.py")
    direct_vm.sender = direct_alice

    contract.add_observation("database connection pool exhausted")
    second = contract.add_observation("cpu temperature nominal")
    assert second["is_novel"] is True


def test_get_closest_returns_best_match(direct_vm, direct_deploy, direct_alice):
    contract = direct_deploy("contracts/anomaly_detection.py")
    direct_vm.sender = direct_alice

    contract.add_observation("payment gateway timeout")
    contract.add_observation("payment gateway timeout")
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