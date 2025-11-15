import pytest
from unittest.mock import Mock, patch

from slam_eval.collection import HuggingFaceCollection, CollectionNotLoadedError


class TestHuggingFaceCollection:
    def test_init(self):
        collection = HuggingFaceCollection(
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
        assert collection.collection_iter is None

    def test_init_with_defaults(self):
        collection = HuggingFaceCollection(
            name="test_collection",
            dataset_name="test_dataset"
        )
        
        assert collection.name == "test_collection"
        assert collection.dataset_name == "test_dataset"
        assert collection.split is None
        assert collection.subset is None

    @patch('slam_eval.collection.datasets.load_dataset')
    def test_load(self, mock_load_dataset):
        # Setup mock dataset that is iterable
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([]))
        mock_load_dataset.return_value = mock_dataset
        
        collection = HuggingFaceCollection(
            name="test_collection",
            dataset_name="test_dataset",
            split="train",
            subset="subset1"
        )
        
        collection.load()
        
        # Verify load_dataset was called with correct parameters
        mock_load_dataset.assert_called_once_with("test_dataset", "subset1", split="train")
        
        # Verify collection attributes are set
        assert collection.collection == mock_dataset
        assert collection.collection_iter is not None

    @patch('slam_eval.collection.datasets.load_dataset')
    def test_load_with_none_values(self, mock_load_dataset):
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([]))
        mock_load_dataset.return_value = mock_dataset
        
        collection = HuggingFaceCollection(
            name="test_collection",
            dataset_name="test_dataset"
        )
        
        collection.load()
        
        # Verify load_dataset was called with None values
        mock_load_dataset.assert_called_once_with("test_dataset", None, split=None)

    @patch('slam_eval.collection.datasets.load_dataset')
    def test_next_returns_eval_case(self, mock_load_dataset):
        # Setup mock dataset with iterator
        mock_item = {"input": "test question", "target": "test answer"}
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([mock_item]))
        mock_load_dataset.return_value = mock_dataset
        
        collection = HuggingFaceCollection(
            name="test_collection",
            dataset_name="test_dataset"
        )
        
        collection.load()
        result = next(collection)
        
        assert result == mock_item

    @patch('slam_eval.collection.datasets.load_dataset')
    def test_next_multiple_items(self, mock_load_dataset):
        # Setup mock dataset with multiple items
        mock_items = [
            {"input": "question 1", "target": "answer 1"},
            {"input": "question 2", "target": "answer 2"},
            {"input": "question 3", "target": "answer 3"}
        ]
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter(mock_items))
        mock_load_dataset.return_value = mock_dataset
        
        collection = HuggingFaceCollection(
            name="test_collection",
            dataset_name="test_dataset"
        )
        
        collection.load()
        
        # Test iterating through all items
        results = []
        for item in mock_items:
            results.append(next(collection))
        
        assert results == mock_items

    def test_next_raises_error_when_not_loaded(self):
        collection = HuggingFaceCollection(
            name="test_collection",
            dataset_name="test_dataset"
        )
        
        with pytest.raises(CollectionNotLoadedError, match="Collection test_collection not loaded"):
            next(collection)

    @patch('slam_eval.collection.datasets.load_dataset')
    def test_len_returns_num_rows(self, mock_load_dataset):
        # Setup mock dataset with num_rows
        mock_split_data = Mock()
        mock_split_data.num_rows = 100
        
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([]))
        mock_dataset.__getitem__ = Mock(return_value=mock_split_data)
        mock_load_dataset.return_value = mock_dataset
        
        collection = HuggingFaceCollection(
            name="test_collection",
            dataset_name="test_dataset",
            split="train"
        )
        
        collection.load()
        result = len(collection)
        
        assert result == 100
        mock_dataset.__getitem__.assert_called_once_with("train")

    def test_len_raises_error_when_not_loaded(self):
        collection = HuggingFaceCollection(
            name="test_collection",
            dataset_name="test_dataset"
        )
        
        with pytest.raises(CollectionNotLoadedError, match="Collection test_collection not loaded"):
            len(collection)

    @patch('slam_eval.collection.datasets.load_dataset')
    def test_iterator_protocol(self, mock_load_dataset):
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([]))
        mock_load_dataset.return_value = mock_dataset
        
        collection = HuggingFaceCollection(
            name="test_collection",
            dataset_name="test_dataset"
        )
        
        # Test that __iter__ returns self
        assert iter(collection) == collection

    @patch('slam_eval.collection.datasets.load_dataset')
    def test_stop_iteration(self, mock_load_dataset):
        # Setup mock dataset that raises StopIteration
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([]))
        mock_load_dataset.return_value = mock_dataset
        
        collection = HuggingFaceCollection(
            name="test_collection",
            dataset_name="test_dataset"
        )
        
        collection.load()
        
        # Should raise StopIteration when no more items
        with pytest.raises(StopIteration):
            next(collection)
