from src.llm_rpc.checkpoints import CheckpointRecord


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
        checkpoint_type=checkpoint_type,
        checkpoint_root_dir=tmp_path,
    )

    # test save_metadata
    assert not record.metadata_path.exists()
    record.save_metadata(
        session_id="sess789",
        base_model="base-model-v1",
        lora_rank=16,
    )
    assert record.metadata_path.exists()

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
    assert record2.metadata["public"] is False
    record2.set_visibility(True)
    assert record2.metadata["public"] is True

    # test delete
    record2.delete()
    assert not checkpoint_dir.exists()
