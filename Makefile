.PHONY: setup dev dev-rust test test-rust test-python test-frontend benchmark-local format lint

SIM_SERVER_URL ?= http://127.0.0.1:8080

setup:
	python -m pip install -e .

dev:
	cd rust && cargo run -p sim-server

dev-rust:
	cd rust && cargo run -p sim-server

test: test-python test-rust test-frontend

test-rust:
	cd rust && cargo test --workspace

test-python:
	python -m pytest -q

test-frontend:
	npm run lint --prefix frontend/observer
	npm exec --prefix frontend/observer -- tsc --noEmit -p frontend/observer/tsconfig.json
	npm run test --prefix frontend/observer
	npm run build --prefix frontend/observer

benchmark-local:
	python -m embodied_ai.cli benchmark --runs 10 --seed-start 1000 --server-url $(SIM_SERVER_URL)

format:
	cd rust && cargo fmt

lint:
	cd rust && cargo clippy --all-targets --all-features
