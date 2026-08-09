"""Run the multi-domain test suite through the snapshot orchestrator."""

from test_script import SnapshotOrchestratorInput, main


if __name__ == "__main__":
    main("orch_pag_snapshot", SnapshotOrchestratorInput)