use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use sim_core::{Action, Direction, Environment, Scenario};

fn survival_room() -> Scenario {
    serde_json::from_str(include_str!(
        "../../../../scenarios/survival_room/scenario.rust.json"
    ))
    .expect("bundled benchmark scenario is valid")
}

fn benchmark_core(c: &mut Criterion) {
    c.bench_function("survival_room/observation", |bench| {
        let env = Environment::new(survival_room(), 42).unwrap();
        bench.iter(|| black_box(env.observe()));
    });
    c.bench_function("survival_room/step_move", |bench| {
        bench.iter_batched(
            || Environment::new(survival_room(), 42).unwrap(),
            |mut env| {
                black_box(env.step(Action::Move {
                    direction: Direction::East,
                }))
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, benchmark_core);
criterion_main!(benches);
