import json

from src.tuft.checkpoints import CheckpointRecord


def test_checkpoint_record(tmp_path):
    # Create a dummy checkpoint record
    training_run_id = "run123"
    checkpoint_id = "ckpt456"
    checkpoint_type = "training"
    checkpoint_dir = tmp_path / training_run_id / checkpoint_id
    checkpoint_dir.mkdir(parents=True)

    # Create from_training_run
    record = CheckpointRecord.from_training_run(
        training_run_id=training_run_id,
        checkpoint_name=checkpoint_id,
        owner_name="default",
        checkpoint_type=checkpoint_type,
        checkpoint_root_dir=tmp_path,
    )

    # test save_metadata
    assert not record.metadata_path.exists()
    record.save_metadata(
        session_id="sess789",
        base_model="base-model-v1",
        lora_rank=16,
        lora_alpha=32,
    )
    assert record.metadata_path.exists()
    assert record.metadata.lora_alpha == 32

    # tinker_path property
    tinker_path = record.tinker_path
    assert tinker_path == f"tinker://{training_run_id}/weights/{checkpoint_id}"
    # other path properties
    assert record.adapter_path == checkpoint_dir / "adapter"
    assert record.optimizer_path == checkpoint_dir / "optimizer"
    assert record.metadata_path == checkpoint_dir / "metadata.json"

    # from_tinker_path
    record2 = CheckpointRecord.from_tinker_path(tinker_path, tmp_path)
    assert record2.checkpoint_id == checkpoint_id
    assert record2.training_run_id == training_run_id
    assert record2.checkpoint_type == checkpoint_type
    assert record2.path == checkpoint_dir
    assert record2.size_bytes == 0
    assert record2.public is False
    assert record2.owner_name == "default"

    # test set_visibility
    assert record2.public is False
    record2.set_visibility(True)
    assert record2.public is True

    # test delete
    record2.delete()
    assert not checkpoint_dir.exists()


def test_saved_target_modules_prefers_adapter_config_and_falls_back_to_metadata(tmp_path):
    record = CheckpointRecord.from_training_run(
        training_run_id="run-targets",
        checkpoint_name="checkpoint-targets",
        owner_name="default",
        checkpoint_type="training",
        checkpoint_root_dir=tmp_path,
    )
    record.save_metadata(
        session_id="session",
        base_model="base-model",
        lora_rank=8,
        target_modules=["q_proj", "v_proj"],
    )

    assert record.saved_target_modules == ["q_proj", "v_proj"]

    record.adapter_path.mkdir()
    (record.adapter_path / "adapter_config.json").write_text(
        json.dumps({"target_modules": ["k_proj", "o_proj"]}),
        encoding="utf-8",
    )
    assert record.saved_target_modules == ["k_proj", "o_proj"]

    (record.adapter_path / "adapter_config.json").write_text("not-json", encoding="utf-8")
    assert record.saved_target_modules == ["q_proj", "v_proj"]
