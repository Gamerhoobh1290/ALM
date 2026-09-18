import copy
import pytest
import torch
from adamlm import bpe_train
from adamlm.model import DecoderTransformer, ModelConfig
from adamlm.training import update, generate
from adamlm.checkpoint import CheckpointManager
from adamlm.data import TinyStoriesStream
from adamlm.storage import StorageBudget, LowDiskSpace


def model():
    return DecoderTransformer(ModelConfig(block_size=8, n_layer=1, n_head=2, n_embd=16))


def test_accumulation_matches_full_batch():
    torch.manual_seed(4)
    full = model()
    micro = copy.deepcopy(full)
    x, y = torch.randint(256, (4, 8)), torch.randint(256, (4, 8))
    a = torch.optim.SGD(full.parameters(), lr=0.01)
    b = torch.optim.SGD(micro.parameters(), lr=0.01)
    loss_a = update(full, a, [(x, y)])
    loss_b = update(micro, b, list(zip(x.chunk(2), y.chunk(2))))
    torch.testing.assert_close(loss_a, loss_b)
    for p, q in zip(full.parameters(), micro.parameters()):
        torch.testing.assert_close(p, q, atol=1e-7, rtol=1e-5)


def test_resume_reproduces_next_update_and_stream(tmp_path):
    torch.manual_seed(8)
    first = model()
    opt = torch.optim.AdamW(first.parameters(), lr=0.001)
    stream = TinyStoriesStream()
    stream.buffer = list(range(256))*4
    x, y = map(torch.tensor, stream.next_batch(2, 8))
    update(first, opt, [(x, y)])
    path = CheckpointManager(tmp_path).save(1, first, opt, stream.state_dict())
    x, y = map(torch.tensor, stream.next_batch(2, 8))
    update(first, opt, [(x, y)])
    second = model()
    opt2 = torch.optim.AdamW(second.parameters(), lr=0.001)
    stream2 = TinyStoriesStream()
    CheckpointManager.load(path, second, opt2, stream2)
    x2, y2 = map(torch.tensor, stream2.next_batch(2, 8))
    assert torch.equal(x, x2) and torch.equal(y, y2)
    update(second, opt2, [(x2, y2)])
    for p, q in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    with pytest.raises(ValueError):
        TinyStoriesStream(split="validation").load_state_dict(stream.state_dict())


def test_future_tokens_cannot_change_earlier_logits():
    net = model().eval()
    x = torch.randint(256, (1, 8))
    y = x.clone()
    y[:, 4:] = (y[:, 4:] + 1) % 256
    torch.testing.assert_close(net(x)[0][:, :4], net(y)[0][:, :4])


def test_checkpoint_reserve_guard(tmp_path):
    budget = StorageBudget(tmp_path, budget_gb=1, minimum_free_gb=0)
    with pytest.raises(LowDiskSpace):
        budget.check(reserve_bytes=2*1024**3)


def test_generation_preserves_training_mode_and_rng():
    net = model()
    state = torch.get_rng_state()
    text = generate(net, "Hi", new_tokens=4)
    assert text.startswith("Hi") and net.training
    assert torch.equal(state, torch.get_rng_state())


# --------------------------------------------------------------------------
# Validation monitor: comparable points, sustained evidence, real protection.
# --------------------------------------------------------------------------

def _policy(**overrides):
    cfg = {"warmup_steps": 50, "minimum_learning_rate": 1e-5,
           "validation_monitor": {"patience": 4, "min_delta": 0.002, "max_regression": 0.08,
                                  "regression_patience": 2, "settled_lr_factor": 2.0, **overrides}}
    return bpe_train.monitor_policy(cfg)


def _fresh(best):
    return {"best_loss": best, "bad_validations": 0, "best_step": 0,
            "regression_streak": 0, "deferred_checks": 0}


def test_monitor_defers_the_peak_learning_rate_excursion():
    """The observed failure: one check at peak LR used to pause the run.

    Replays the real shape of a stage (2.73 baseline, excursion to ~3.9 while
    the schedule is at peak, recovery as it anneals) and asserts no pause.
    """
    policy = _policy()
    monitor = _fresh(2.733918)
    excursion = [(50, 1e-4, 3.871392), (100, 9.98e-5, 3.972500), (650, 8.10e-5, 4.005000),
                 (1250, 4.01e-5, 3.491000), (1550, 2.23e-5, 2.975100)]
    for step, lr, loss in excursion:
        monitor, action, _ = bpe_train.validation_verdict(monitor, loss, step, lr, policy)
        assert action == "deferred", f"step {step} judged at lr {lr:.2e}"
    assert monitor["deferred_checks"] == len(excursion)
    assert monitor["bad_validations"] == 0
    # Recovery below the baseline is still recognised as a real improvement.
    monitor, action, _ = bpe_train.validation_verdict(monitor, 2.700000, 2000, 1.0e-5, policy)
    assert action == "improved" and monitor["best_loss"] == 2.700000


def test_monitor_still_pauses_a_genuine_regression_at_a_settled_point():
    """Protection is intact: repeated breaches at settled LR pause the run."""
    policy = _policy()
    monitor = _fresh(2.500000)
    monitor, first, _ = bpe_train.validation_verdict(monitor, 3.000000, 1900, 1.2e-5, policy)
    assert first == "bad" and monitor["regression_streak"] == 1
    monitor, second, detail = bpe_train.validation_verdict(monitor, 3.100000, 1950, 1.1e-5, policy)
    assert second == "pause"
    assert "2 consecutive settled checks" in detail


def test_monitor_requires_the_breach_to_be_sustained_not_a_single_spike():
    policy = _policy()
    monitor = _fresh(2.500000)
    monitor, _, _ = bpe_train.validation_verdict(monitor, 3.000000, 1900, 1.2e-5, policy)
    # A settled check back inside the threshold clears the streak.
    monitor, action, _ = bpe_train.validation_verdict(monitor, 2.540000, 1950, 1.1e-5, policy)
    assert action == "bad" and monitor["regression_streak"] == 0
    monitor, action, _ = bpe_train.validation_verdict(monitor, 3.000000, 2000, 1.0e-5, policy)
    assert action == "bad" and monitor["regression_streak"] == 1


def test_monitor_still_pauses_on_sustained_failure_to_improve():
    """The patience path survives: four settled checks without improvement."""
    policy = _policy()
    monitor = _fresh(2.500000)
    for index in range(3):
        monitor, action, _ = bpe_train.validation_verdict(
            monitor, 2.520000, 1800 + index * 50, 1.2e-5, policy)
        assert action == "bad"
    monitor, action, detail = bpe_train.validation_verdict(monitor, 2.520000, 1950, 1.1e-5, policy)
    assert action == "pause" and "4 settled checks" in detail


def test_monitor_warmup_floor_is_never_below_the_schedule_warmup():
    assert _policy(warmup_steps=0)["warmup_steps"] == 50
    assert _policy(warmup_steps=500)["warmup_steps"] == 500
    # A settled-LR factor below 1 cannot make the gate stricter than the floor.
    assert _policy(settled_lr_factor=0.1)["settled_lr"] == 1e-5


def test_monitor_retuning_does_not_strand_an_existing_checkpoint():
    """Stopping policy is not lineage: changing it must not block resume."""
    base = {"run_dir": "x", "seed": 1, "validation_monitor": {"patience": 4}}
    retuned = {"run_dir": "x", "seed": 1,
               "validation_monitor": {"patience": 4, "regression_patience": 2}}
    assert bpe_train.protocol_config(base) == bpe_train.protocol_config(retuned)
