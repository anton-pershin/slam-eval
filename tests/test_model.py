import pytest
from unittest.mock import Mock, patch

from rally.llm import Llm
from slam_eval.model import LlmViaOpenAiApi
from slam_eval.collections.text_generation import TextGenerationInput


class TestLlmViaOpenAiApi:
    def test_init(self):
        mock_llm = Mock(spec=Llm)
        mock_llm.url = "http://test-url.com"
        mock_llm.authorization = "Bearer test-token"
        mock_llm.model = "test-model"
        
        model = LlmViaOpenAiApi("test_model", mock_llm)
        
        assert model.name == "test_model"
        assert model.llm == mock_llm

    @patch('slam_eval.model.request_based_on_message_history')
    def test_predict_returns_content(self, mock_request):
        # Setup mock LLM
        mock_llm = Mock(spec=Llm)
        mock_llm.url = "http://test-url.com"
        mock_llm.authorization = "Bearer test-token"
        mock_llm.model = "test-model"
        
        # Setup mock response
        mock_request.return_value = {
            "role": "assistant",
            "content": "This is the model's response"
        }
        
        model = LlmViaOpenAiApi("test_model", mock_llm)
        input_data = TextGenerationInput(
            system_prompt=None,
            user_prompt="What is 2+2?"
        )
        result = model.predict(input_data)
        
        assert result == "This is the model's response"

    @patch('slam_eval.model.request_based_on_message_history')
    def test_predict_calls_request_with_correct_parameters(self, mock_request):
        # Setup mock LLM
        mock_llm = Mock(spec=Llm)
        mock_llm.url = "http://test-url.com"
        mock_llm.authorization = "Bearer test-token"
        mock_llm.model = "gpt-4"
        
        # Setup mock response
        mock_request.return_value = {
            "role": "assistant",
            "content": "Response content"
        }
        
        model = LlmViaOpenAiApi("test_model", mock_llm)
        input_data = TextGenerationInput(
            system_prompt=None,
            user_prompt="What is the capital of France?"
        )
        model.predict(input_data)
        
        # Verify the request was called with correct parameters
        mock_request.assert_called_once_with(
            llm_server_url="http://test-url.com",
            message_history=[{
                "role": "user",
                "content": "What is the capital of France?"
            }],
            authorization="Bearer test-token",
            model="gpt-4"
        )

    @patch('slam_eval.model.request_based_on_message_history')
    def test_predict_with_different_inputs(self, mock_request):
        mock_llm = Mock(spec=Llm)
        mock_llm.url = "http://test-url.com"
        mock_llm.authorization = "Bearer test-token"
        mock_llm.model = "test-model"
        
        # Test with different response content
        mock_request.return_value = {
            "role": "assistant",
            "content": "42"
        }
        
        model = LlmViaOpenAiApi("math_model", mock_llm)
        input_data = TextGenerationInput(
            system_prompt=None,
            user_prompt="Calculate 6*7"
        )
        result = model.predict(input_data)
        
        assert result == "42"

    @patch('slam_eval.model.request_based_on_message_history')
    def test_predict_message_format(self, mock_request):
        mock_llm = Mock(spec=Llm)
        mock_llm.url = "http://test-url.com"
        mock_llm.authorization = "Bearer test-token"
        mock_llm.model = "test-model"
        
        mock_request.return_value = {
            "role": "assistant",
            "content": "Test response"
        }
        
        model = LlmViaOpenAiApi("test_model", mock_llm)
        input_data = TextGenerationInput(
            system_prompt="You are a helpful assistant",
            user_prompt="Hello, world!"
        )
        model.predict(input_data)
        
        # Check that the message history has the correct format
        call_args = mock_request.call_args
        message_history = call_args[1]['message_history']
        
        assert len(message_history) == 2
        assert message_history[0]["role"] == "system"
        assert message_history[0]["content"] == "You are a helpful assistant"
        assert message_history[1]["role"] == "user"
        assert message_history[1]["content"] == "Hello, world!"

    @patch('slam_eval.model.request_based_on_message_history')
    def test_predict_with_only_user_prompt(self, mock_request):
        mock_llm = Mock(spec=Llm)
        mock_llm.url = "http://test-url.com"
        mock_llm.authorization = "Bearer test-token"
        mock_llm.model = "test-model"
        
        mock_request.return_value = {
            "role": "assistant",
            "content": "Test response"
        }
        
        model = LlmViaOpenAiApi("test_model", mock_llm)
        input_data = TextGenerationInput(
            system_prompt=None,
            user_prompt="Hello, world!"
        )
        model.predict(input_data)
        
        # Check that only user message is included when system_prompt is None
        call_args = mock_request.call_args
        message_history = call_args[1]['message_history']
        
        assert len(message_history) == 1
        assert message_history[0]["role"] == "user"
        assert message_history[0]["content"] == "Hello, world!"
