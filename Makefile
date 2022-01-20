PYTHON = python3
PIP = pip3

.DEFAULT_GOAL = run

build:
	bash scripts/build.sh

run:
	bash scripts/run.sh $(filter-out $@, $(MAKECMDGOALS))