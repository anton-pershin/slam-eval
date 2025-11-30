import pytest
from unittest.mock import Mock, patch

from slam_eval.collections.base import CollectionNotLoadedError
from slam_eval.collections.text_generation import BigBenchHard


class TestBigBenchHard:
    def test_init(self):
        collection = BigBenchHard(
            name="test_collection",
            dataset_name="test_dataset",
            split="train",
            subset="subset1"
        )
        
        assert collection.name == "test_collection"
        assert collection.dataset_name == "test_dataset"
        assert collection.split == "train"
        assert collection.subset == "subset1"
        assert collection.collection is None
        assert collection.collection_len is None

    @patch('slam_eval.collections.text_generation.datasets.load_dataset')
    def test_load(self, mock_load_dataset):
        # Setup mock dataset that is iterable
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([]))
        mock_dataset.num_rows = 100
        mock_load_dataset.return_value = mock_dataset
        
        collection = BigBenchHard(
            name="test_collection",
            dataset_name="test_dataset",
            split="train",
            subset="subset1"
        )
        
        collection.load()
        
        # Verify load_dataset was called with correct parameters
        mock_load_dataset.assert_called_once_with("test_dataset", "subset1", split="train")
        
        # Verify collection attributes are set
        assert collection.collection is not None
        assert collection.collection_len == 100

    @patch('slam_eval.collections.text_generation.datasets.load_dataset')
    def test_next_returns_eval_case(self, mock_load_dataset):
        # Setup mock dataset with iterator
        mock_item = {"input": "test question", "target": "test answer"}
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([mock_item]))
        mock_dataset.num_rows = 1
        mock_load_dataset.return_value = mock_dataset
        
        collection = BigBenchHard(
            name="test_collection",
            dataset_name="test_dataset",
            split="train",
            subset="subset1"
        )
        
        collection.load()
        result = next(collection)
        
        # Verify the result has the correct structure
        assert "x" in result
        assert "y_true" in result
        assert result["x"]["system_prompt"] is None
        assert result["x"]["user_prompt"] == "test question"
        assert result["y_true"] == "test answer"

    @patch('slam_eval.collections.text_generation.datasets.load_dataset')
    def test_next_multiple_items(self, mock_load_dataset):
        # Setup mock dataset with multiple items
        mock_items = [
            {"input": "question 1", "target": "answer 1"},
            {"input": "question 2", "target": "answer 2"},
            {"input": "question 3", "target": "answer 3"}
        ]
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter(mock_items))
        mock_dataset.num_rows = 3
        mock_load_dataset.return_value = mock_dataset
        
        collection = BigBenchHard(
            name="test_collection",
            dataset_name="test_dataset",
            split="train",
            subset="subset1"
        )
        
        collection.load()
        
        # Test iterating through all items
        results = []
        for _ in mock_items:
            results.append(next(collection))
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["x"]["user_prompt"] == f"question {i+1}"
            assert result["y_true"] == f"answer {i+1}"
            assert result["x"]["system_prompt"] is None

    @patch('slam_eval.collections.text_generation.datasets.load_dataset')
    def test_key_mapping_transformation(self, mock_load_dataset):
        # Test specifically for the input->x and target->y_true mapping
        mock_item = {"input": "What is 2+2?", "target": "4"}
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([mock_item]))
        mock_dataset.num_rows = 1
        mock_load_dataset.return_value = mock_dataset
        
        collection = BigBenchHard(
            name="test_collection",
            dataset_name="test_dataset",
            split="train",
            subset="subset1"
        )
        
        collection.load()
        result = next(collection)
        
        # Verify exact key mapping and structure
        assert result["x"]["user_prompt"] == "What is 2+2?"
        assert result["x"]["system_prompt"] is None
        assert result["y_true"] == "4"
        assert len(result) == 2  # Only x and y_true keys should be present
        assert len(result["x"]) == 2  # Only system_prompt and user_prompt

    def test_next_raises_error_when_not_loaded(self):
        collection = BigBenchHard(
            name="test_collection",
            dataset_name="test_dataset",
            split="train",
            subset="subset1"
        )
        
        with pytest.raises(CollectionNotLoadedError, match="Collection test_collection not loaded"):
            next(collection)

    @patch('slam_eval.collections.text_generation.datasets.load_dataset')
    def test_len_returns_num_rows(self, mock_load_dataset):
        # Setup mock dataset with num_rows attribute
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([]))
        mock_dataset.num_rows = 100
        mock_load_dataset.return_value = mock_dataset
        
        collection = BigBenchHard(
            name="test_collection",
            dataset_name="test_dataset",
            split="train",
            subset="subset1"
        )
        
        collection.load()
        result = len(collection)
        
        assert result == 100

    def test_len_raises_error_when_not_loaded(self):
        collection = BigBenchHard(
            name="test_collection",
            dataset_name="test_dataset",
            split="train",
            subset="subset1"
        )
        
        with pytest.raises(TypeError, match="Collection not loaded. Call load\\(\\) first."):
            len(collection)

    @patch('slam_eval.collections.text_generation.datasets.load_dataset')
    def test_iterator_protocol(self, mock_load_dataset):
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([]))
        mock_dataset.num_rows = 0
        mock_load_dataset.return_value = mock_dataset
        
        collection = BigBenchHard(
            name="test_collection",
            dataset_name="test_dataset",
            split="train",
            subset="subset1"
        )
        
        # Test that __iter__ returns self
        assert iter(collection) == collection

    @patch('slam_eval.collections.text_generation.datasets.load_dataset')
    def test_stop_iteration(self, mock_load_dataset):
        # Setup mock dataset that raises StopIteration
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([]))
        mock_dataset.num_rows = 0
        mock_load_dataset.return_value = mock_dataset
        
        collection = BigBenchHard(
            name="test_collection",
            dataset_name="test_dataset",
            split="train",
            subset="subset1"
        )
        
        collection.load()
        
        # Should raise StopIteration when no more items
        with pytest.raises(StopIteration):
            next(collection)
