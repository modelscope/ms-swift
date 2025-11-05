import re
import textwrap
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, List

import torch

from swift.llm import PtEngine, RequestConfig, Template, to_device
from swift.utils import get_logger

if TYPE_CHECKING:
    from swift.llm.infer.protocol import ChatCompletionResponse

logger = get_logger()


class DefaultRMPlugin:
    """
    Default Reward Model Plugin

    This class implements the default processing logic for reward models.
    It assumes that `self.model` is a classification model with a value head(output dimmension 1).
    The first logits value from the model's output is used as the reward score.
    """

    def __init__(self, model, template):
        self.model = model
        self.template: Template = template

    def __call__(self, inputs, **kwargs):
        batched_inputs = [self.template.encode(deepcopy(infer_request)) for infer_request in inputs]
        reward_inputs = to_device(self.template.data_collator(batched_inputs), self.model.device)

        with torch.inference_mode():
            return self.model(**reward_inputs).logits[:, 0]


class GenRMPlugin(DefaultRMPlugin):

    def __init__(self, model, template):
        """
        Generative Reward Model Plugin Example.

        This method sets up the reward model plugin by initializing the PtEngine for efficient inference,
        configuring the request parameters, and defining the system prompt that guides the reward model in
        evaluating responses.

        Args:
            model (torch.nn.Module): The generative reward model.
            template (Template): The template used for encoding input data.
    """

        super().__init__(model, template)
        # initilize PTEngine to infer
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)  # 0: no limit
        self.request_config = RequestConfig()  # customise your request config here
        self.system = textwrap.dedent("""
            Based on the dialogue history, analyze in detail whether the model's response is accurate, complete, and relevant.
            Assign a reward score between 0 and 1, where 0 indicates completely incorrect and 1 indicates fully correct.
            Before finishing your response, please assign a reward using the following format:

            Reward: {reward}

            For example:
            Reward: 0.85
        """)  # noqa

    def __call__(self, inputs, **kwargs):
        """
        Compute reward scores for the provided inputs.

        This method processes each input by converting dialogue messages into a query, sending the query to the
        reward model for inference, and extracting the reward scores from the model's responses. The final reward
        for each input is the average of all extracted scores.
        Args:
            inputs (List[Dict]): A list of input requests. Each input request is a dictionary containing:
                - 'messages' (List[Dict]): messages from the training model. Each message dictionary includes:
                    - 'role' (str): The role of the speaker (e.g., 'user', 'assistant').
                    - 'content' (str): The content of the message.
                - Additional dataset columns as key-value pairs (e.g., 'solutions', 'images').
        Returns:
            torch.Tensor: A tensor containing the average reward scores for each input. The tensor has a shape of (N,),
            where N is the number of input requests.
        """

        rm_inputs = self.prepare_rm_inputs(inputs)
        results = self.engine.infer(rm_inputs, self.request_config, use_tqdm=False)
        rewards = self.compute_rewards(results)
        return torch.tensor(rewards, dtype=torch.float32)

    def prepare_rm_inputs(self, inputs: List[Dict]) -> List[Dict]:
        """
        Prepare inputs for the reward model by converting messages into queries.

        Args:
            inputs (List[Dict]): A list of input requests.

        Returns:
            List[Dict]: Processed inputs for the reward model.
        """
        rm_inputs = []
        for idx, infer_request in enumerate(inputs):
            # Deep copy to prevent modification of original input
            rm_infer_request = deepcopy(infer_request)

            # Extract and convert messages to a single query string
            messages = rm_infer_request.get('messages')
            query = self.messages_to_query(messages)

            # Construct new messages tailored for the reward model
            rm_messages = [{'role': 'system', 'content': self.system}, {'role': 'user', 'content': query}]

            # Update the messages in the reward infer request
            rm_infer_request['messages'] = rm_messages
            rm_inputs.append(rm_infer_request)
        return rm_inputs

    @staticmethod
    def extract_reward(model_output: str) -> float:
        """
        Extract the reward score from the model's output.

        Args:
            model_output (str): The model's output string, expected to follow the format "Reward: {reward}".

        Returns:
            float: The extracted reward score.

        Raises:
            ValueError: If the reward score cannot be extracted or the format is incorrect.
        """
        match = re.search(r'Reward:\s*([0-1](?:\.\d+)?)', model_output)
        if match:
            return float(match.group(1))
        else:
            logger.warning("Unable to extract reward score from the model's output, set reward to 0")
            return None

    @staticmethod
    def messages_to_query(messages):
        """
        Compress a list of message dictionaries into a single query string.

        Args:
            messages (list[dict]): A list of message dictionaries, each containing:
                - 'role' (str): The role of the speaker (e.g., 'user', 'assistant').
                - 'content' (str): The content of the message.

        Returns:
            str: A single string that concatenates all messages in a formatted manner.

        Example:
            >>> messages = [
            ...     {'role': 'user', 'content': 'Hello, how are you?'},
            ...     {'role': 'assistant', 'content': 'I am fine, thank you! How can I assist you today?'},
            ...     {'role': 'user', 'content': 'Can you help me with my homework?'}
            ... ]
            >>> print(messages_to_query(messages))
            User: Hello, how are you?
            Assistant: I am fine, thank you! How can I assist you today?
            User: Can you help me with my homework?
        """
        # Initialize an empty list to hold formatted messages
        formatted_messages = []

        # Define a mapping for role capitalization if needed
        role_mapping = {
            'user': 'User',
            'assistant': 'Assistant',
            'system': 'System'
            # Add more roles here as needed
        }

        for idx, message in enumerate(messages):
            if not isinstance(message, dict):
                raise TypeError(f'Each message must be a dictionary. Found {type(message)} at index {idx}.')

            # Extract 'role' and 'content' from each message
            role = message.get('role')
            content = message.get('content')
            if not content:
                continue

            # Capitalize the role using the mapping, default to capitalized original role
            role_formatted = role_mapping.get(role.lower(), role.capitalize())

            # Append the formatted message to the list
            formatted_messages.append(f'{role_formatted}: {content}')

        # Join all formatted messages with newline characters
        query = '\n'.join(formatted_messages)

        return query

    def compute_rewards(self, results: List['ChatCompletionResponse']) -> List[float]:
        """
        Compute average reward scores from the reward model's outputs.

        Args:
            results (List['ChatCompletionResponse']): A list of results from the reward model.

        Returns:
            List[float]: A list of average reward scores.
        """
        rewards = []
        for idx, output in enumerate(results):
            try:
                cur_rewards = []
                for choice in output.choices:
                    response = choice.message.content
                    reward = self.extract_reward(response)
                    cur_rewards.append(reward)
                cur_rewards = [r for r in cur_rewards if r is not None]
                if cur_rewards:
                    average_reward = sum(cur_rewards) / len(cur_rewards)
                else:
                    average_reward = 0.0
                    logger.warning('No valid rewards extracted. Assigning reward score of 0.0.')

                rewards.append(average_reward)
            except Exception as e:
                logger.error(f'Error computing reward: {e}')
                rewards.append(0.0)  # Assign default reward score on failure
        return rewards


rm_plugins = {
    'default': DefaultRMPlugin,
    'genrm': GenRMPlugin,
}
