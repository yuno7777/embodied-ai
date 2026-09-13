# Local performance notes

These are observations from one local Windows development machine, not portable capacity claims. Repeat the command on the target machine before making a performance decision.

## 2026-09-13: initial parallel smoke measurement

Command:

```powershell
python -m embodied_ai.cli benchmark-scale --provider scripted --runs 8 --workers 1,8 --server-url http://127.0.0.1:8080
```

The authoritative Rust server ran as a separate local process. The benchmark used eight catalog `survival_room` episodes at each level.

| Workers | Simulation steps/sec | Mean control elapsed ms | Mean step latency ms |
| ---: | ---: | ---: | ---: |
| 1 | 8602.52 | 1163.5 | 24.16 |
| 8 | 8381.88 | 7056.0 | 105.81 |

Simulation throughput stayed broadly flat while control latency increased substantially at eight workers. This points to Python/HTTP/control-path contention in this small run; it does not establish a Rust-core scaling limit. The 32- and 64-worker levels have not yet been measured on this machine.
