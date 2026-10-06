"""Launcher safety with a fake `vastai` CLI: no network, no money."""

import pytest

from ygo_zero.infra import vast


class FakeVast:
    def __init__(self, ready_ok=(False, True)):
        self.calls, self.instances, self.ready = [], {}, list(ready_ok)
        self.next_id = 100

    def __call__(self, *args, raw=True, timeout=120):
        self.calls.append(args)
        match args[:2]:
            case ("show", "user"):
                return {"credit": 5.0}
            case ("show", "instances"):
                return [{"id": i, "label": lab} for i, lab in self.instances.items()]
            case ("search", "offers"):
                return [
                    {"id": 1, "machine_id": 11, "dph_total": 0.05},
                    {"id": 2, "machine_id": 22, "dph_total": 0.06},
                ]
            case ("create", "instance"):
                self.next_id += 1
                self.instances[self.next_id] = args[args.index("--label") + 1]
                return {"new_contract": self.next_id}
            case ("destroy", "instance"):
                assert "-y" in args, "destroy without -y silently aborts"
                self.instances.pop(int(args[2]), None)
                return ""
        raise AssertionError(args)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    f = FakeVast()
    monkeypatch.setattr(vast, "cli", f)
    monkeypatch.setattr(vast, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(vast, "BAD_MACHINES", tmp_path / "bad.json")
    monkeypatch.setattr(vast, "ROOT", tmp_path)
    monkeypatch.setattr(vast, "ALLOWED_FILE", tmp_path / "allowed.txt")
    monkeypatch.setattr(vast.time, "sleep", lambda s: None)
    return f


def test_failed_host_is_replaced_and_everything_destroyed(fake, monkeypatch):
    def wait_ready(iid, timeout_s=900):
        if not fake.ready.pop(0):
            raise vast.HostFailed("broken GPU")
        raise RuntimeError("ssh died")  # second host: fail later in the job

    monkeypatch.setattr(vast, "wait_ready", wait_ready)
    with pytest.raises(RuntimeError, match="ssh died"):
        vast.run_job(vast.JobSpec("t", "true"))
    created = [c for c in fake.calls if c[:2] == ("create", "instance")]
    assert [c[2] for c in created] == ["1", "2"]  # second attempt avoided machine 11
    assert fake.instances == {}  # both destroyed
    assert (vast.LEDGER).read_text().count("\n") == 1


def test_refuses_when_worst_case_exceeds_credit(fake):
    with pytest.raises(vast.VastError, match="worst-case"):
        vast.run_job(vast.JobSpec("t", "true", hours_cap=200))
    assert not [c for c in fake.calls if c[:2] == ("create", "instance")]


def test_ssh_target_prefers_direct():
    direct = {
        "public_ipaddr": "1.2.3.4",
        "ports": {"22/tcp": [{"HostPort": "51234"}]},
        "ssh_host": "ssh9.vast.ai",
        "ssh_port": "12345",
    }
    assert vast._ssh_target(direct) == ("1.2.3.4", 51234)
    proxy = {"ssh_host": "ssh9.vast.ai", "ssh_port": "12345"}
    assert vast._ssh_target(proxy) == ("ssh9.vast.ai", 12345)


class FakeSSH:
    """Fake subprocess.run for relay tests: serves cat <rdir>/<file> from a dict."""

    def __init__(self, files):
        self.files = files

    def __call__(self, cmd, **kwargs):
        class Out:
            pass

        out = Out()
        last = cmd[-1]  # "cat <rdir>/<filename> 2>/dev/null"
        name = "ALERT" if "/ALERT " in last or last.rstrip().endswith("/ALERT") else (
            "PROGRESS" if "/PROGRESS" in last else last.split("/")[-1].split()[0]
        )
        out.stdout = self.files.get(name, "")
        out.returncode = 0
        return out


def test_relay_alerts_prefix_and_incremental(tmp_path, monkeypatch):
    import subprocess

    log = tmp_path / "launcher.log"
    log.write_text("")
    fake = FakeSSH({"ALERT": "boom 1\nALERT boom 2\n"})
    monkeypatch.setattr(vast.subprocess, "run", fake)
    seen = vast.relay_alerts("h", 22, "runs/t", log, 0)
    assert seen == 2
    assert log.read_text().splitlines() == ["ALERT boom 1", "ALERT boom 2"]
    fake.files["ALERT"] += "boom 3\n"
    seen = vast.relay_alerts("h", 22, "runs/t", log, seen)
    assert seen == 3
    assert log.read_text().splitlines()[-1] == "ALERT boom 3"


def test_relay_progress_prefixed_and_missing_is_normal(tmp_path, monkeypatch):
    log = tmp_path / "launcher.log"
    log.write_text("")
    fake = FakeSSH({"PROGRESS": "[eval it 10] wr 0.6\n[eval it 20] wr 0.7\n"})
    monkeypatch.setattr(vast.subprocess, "run", fake)
    seen = vast.relay_alerts("h", 22, "runs/t", log, 0, "PROGRESS", "PROGRESS ")
    assert seen == 2
    assert log.read_text().splitlines() == [
        "PROGRESS [eval it 10] wr 0.6",
        "PROGRESS [eval it 20] wr 0.7",
    ]
    # missing file is normal (other job types): no error, seen unchanged
    fake.files.pop("PROGRESS")
    seen2 = vast.relay_alerts("h", 22, "runs/t", log, seen, "PROGRESS", "PROGRESS ")
    assert seen2 == seen
    assert log.read_text().splitlines() == [
        "PROGRESS [eval it 10] wr 0.6",
        "PROGRESS [eval it 20] wr 0.7",
    ]
