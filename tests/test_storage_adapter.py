import json
import tempfile
import os
from unittest.mock import Mock

import pytest

from slam_eval.storage_adapter import LocalJsonlAdapter
from slam_eval.model import Model
from slam_eval.collection import EvalCaseCollection


class TestLocalJsonlAdapter:
    def test_init(self):
        adapter = LocalJsonlAdapter("/path/to/file.jsonl")
        assert adapter.path_to_jsonl == "/path/to/file.jsonl"

    def test_save_result_dict_creates_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = os.path.join(temp_dir, "test_results.jsonl")
            adapter = LocalJsonlAdapter(jsonl_path)
            
            test_dict = {
                "eval_id": "test_eval_123",
                "model": "test_model",
                "scores": [1, 0, 1]
            }
            
            adapter._save_result_dict(test_dict)
            
            # Check file was created
            assert os.path.exists(jsonl_path)
            
            # Check content
            with open(jsonl_path, 'r') as f:
                content = f.read().strip()
                loaded_dict = json.loads(content)
                assert loaded_dict == test_dict

    def test_save_result_dict_appends_to_existing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = os.path.join(temp_dir, "test_results.jsonl")
            adapter = LocalJsonlAdapter(jsonl_path)
            
            # Save first result
            first_dict = {"eval_id": "test_1", "score": 0.8}
            adapter._save_result_dict(first_dict)
            
            # Save second result
            second_dict = {"eval_id": "test_2", "score": 0.9}
            adapter._save_result_dict(second_dict)
            
            # Check both results are in file
            with open(jsonl_path, 'r') as f:
                lines = f.read().strip().split('\n')
                assert len(lines) == 2
                assert json.loads(lines[0]) == first_dict
                assert json.loads(lines[1]) == second_dict

    def test_save_integration_with_base_class(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = os.path.join(temp_dir, "test_results.jsonl")
            adapter = LocalJsonlAdapter(jsonl_path)
            
            # Mock model and collection
            mock_model = Mock(spec=Model)
            mock_model.name = "test_model"
            
            mock_collection = Mock(spec=EvalCaseCollection)
            mock_collection.name = "test_collection"
            
            # Call the save method from base class
            scores = [1, 0, 1]
            model_answers = ["answer1", "answer2", "answer3"]
            
            adapter.save(
                model=mock_model,
                eval_case_collection=mock_collection,
                scores=scores,
                model_answers=model_answers,
                custom_field="custom_value"
            )
            
            # Check file content
            with open(jsonl_path, 'r') as f:
                content = f.read().strip()
                result_dict = json.loads(content)
                
                assert result_dict["model"] == "test_model"
                assert result_dict["eval_case_collection"] == "test_collection"
                assert result_dict["scores"] == scores
                assert result_dict["model_answers"] == model_answers
                assert result_dict["custom_field"] == "custom_value"
                assert "eval_id" in result_dict
                assert "timestamp" in result_dict

    def test_save_multiple_calls_append_correctly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = os.path.join(temp_dir, "test_results.jsonl")
            adapter = LocalJsonlAdapter(jsonl_path)
            
            # Mock objects
            mock_model = Mock(spec=Model)
            mock_model.name = "test_model"
            mock_collection = Mock(spec=EvalCaseCollection)
            mock_collection.name = "test_collection"
            
            # Save multiple results
            for i in range(3):
                adapter.save(
                    model=mock_model,
                    eval_case_collection=mock_collection,
                    scores=[i],
                    model_answers=[f"answer_{i}"]
                )
            
            # Check all results are saved
            with open(jsonl_path, 'r') as f:
                lines = f.read().strip().split('\n')
                assert len(lines) == 3
                
                for i, line in enumerate(lines):
                    result_dict = json.loads(line)
                    assert result_dict["scores"] == [i]
                    assert result_dict["model_answers"] == [f"answer_{i}"]
