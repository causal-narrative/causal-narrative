"""LLM client for SiliconFlow API with multi-key concurrent processing."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from loguru import logger
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from tqdm import tqdm


@dataclass
class LLMResponse:
    """Response from LLM API call.

    Attributes:
        content: Generated text content
        prompt_tokens: Number of tokens in prompt
        completion_tokens: Number of tokens in completion
        total_tokens: Total tokens used
        model: Model name used
        finish_reason: Reason for completion finish
        raw_response: Raw API response object
    """

    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    finish_reason: str
    raw_response: Optional[Any] = None


class LLMClient:
    """Client for SiliconFlow LLM API with multi-key support.

    Supports concurrent requests across multiple API keys with automatic
    rate limiting, retry logic, and cost tracking.

    Attributes:
        base_url: API base URL
        model_name: Name of model to use
        api_keys: List of API keys for rotation
        max_workers: Maximum concurrent workers
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        temperature: Sampling temperature
        total_cost: Total API cost accumulated
        total_tokens: Total tokens used
    """

    # Pricing per 1M tokens (example - adjust to actual pricing)
    PRICING = {
        "input": 0.14,  # USD per 1M input tokens
        "output": 0.28,  # USD per 1M output tokens
    }

    def __init__(
        self,
        api_keys: List[str],
        model_name: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://api.siliconflow.cn/v1",
        max_workers: int = 4,
        timeout: int = 60,
        max_retries: int = 3,
        temperature: float = 0.0,
    ):
        """Initialize LLM client.

        Args:
            api_keys: List of API keys for concurrent processing
            model_name: Model identifier
            base_url: API base URL
            max_workers: Maximum concurrent workers
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            temperature: Sampling temperature (0.0 = deterministic)
        """
        if not api_keys:
            raise ValueError("At least one API key is required")

        self.base_url = base_url
        self.model_name = model_name
        self.api_keys = api_keys
        self.max_workers = min(max_workers, len(api_keys))
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature

        # Cost tracking
        self.total_cost = 0.0
        self.total_tokens = 0

        # Create clients for each key
        self.clients = [
            OpenAI(api_key=key, base_url=base_url, timeout=timeout) for key in api_keys
        ]

        logger.info(
            f"Initialized LLM client with {len(api_keys)} keys, "
            f"max_workers={self.max_workers}, model={model_name}"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def _make_request(
        self,
        client: OpenAI,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """Make a single API request with retry logic.

        Args:
            client: OpenAI client instance
            messages: List of message dictionaries
            **kwargs: Additional parameters for API call

        Returns:
            LLMResponse object

        Raises:
            Exception: If request fails after retries
        """
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                **kwargs,
            )

            # Extract response data
            choice = response.choices[0]
            content = choice.message.content or ""

            # Extract token usage
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0

            # Update cost tracking
            self._update_cost(prompt_tokens, completion_tokens)

            return LLMResponse(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                model=response.model,
                finish_reason=choice.finish_reason,
                raw_response=response,
            )

        except Exception as e:
            # Never log API keys
            error_msg = str(e).replace(
                client.api_key[:20] if len(client.api_key) > 20 else client.api_key, "***"
            )
            logger.error(f"API request failed: {error_msg}")
            raise

    def _update_cost(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Update cost tracking.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
        """
        input_cost = (prompt_tokens / 1_000_000) * self.PRICING["input"]
        output_cost = (completion_tokens / 1_000_000) * self.PRICING["output"]

        self.total_cost += input_cost + output_cost
        self.total_tokens += prompt_tokens + completion_tokens

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a single response.

        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for API call

        Returns:
            LLMResponse object
        """
        logger.debug(f"Generating single LLM response for prompt: {prompt[:50]}...")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Use first client for single requests
        response = self._make_request(self.clients[0], messages, **kwargs)
        logger.debug(f"LLM response generated: {response.total_tokens} tokens")
        return response

    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[LLMResponse]:
        """Generate responses for multiple prompts concurrently.

        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt for all requests
            show_progress: Show progress bar
            **kwargs: Additional parameters for API calls

        Returns:
            List of LLMResponse objects in same order as prompts
        """
        if not prompts:
            return []

        logger.info(f"Starting batch LLM generation for {len(prompts)} prompts using {self.max_workers} workers")
        
        # Prepare messages for each prompt
        all_messages = []
        for prompt in prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            all_messages.append(messages)

        # Process concurrently with round-robin key assignment
        results = [None] * len(prompts)
        futures_to_idx = {}
        error_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            for idx, messages in enumerate(all_messages):
                client = self.clients[idx % len(self.clients)]
                future = executor.submit(self._make_request, client, messages, **kwargs)
                futures_to_idx[future] = idx

            # Collect results with progress bar
            iterator = as_completed(futures_to_idx)
            if show_progress:
                iterator = tqdm(iterator, total=len(prompts), desc="Generating")

            for future in iterator:
                idx = futures_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Failed to process prompt {idx}: {e}")
                    error_count += 1
                    # Return a placeholder error response
                    results[idx] = LLMResponse(
                        content=f"ERROR: {str(e)}",
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        model=self.model_name,
                        finish_reason="error",
                    )

        total_tokens = sum(r.total_tokens for r in results if r.total_tokens > 0)
        logger.info(f"Completed batch LLM generation: {len(prompts)} prompts, {error_count} errors, {total_tokens} total tokens, cost=${self.total_cost:.4f}")
        return results

    def batch_generate_with_metadata(
        self,
        items: List[Tuple[int, str]],
        system_prompt: Optional[str] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[Tuple[int, LLMResponse]]:
        """Generate responses with metadata preserved.

        Useful when processing items with IDs or indices that need to be preserved.

        Args:
            items: List of (id, prompt) tuples
            system_prompt: Optional system prompt for all requests
            show_progress: Show progress bar
            **kwargs: Additional parameters for API calls

        Returns:
            List of (id, LLMResponse) tuples
        """
        if not items:
            return []

        ids, prompts = zip(*items)
        responses = self.batch_generate(prompts, system_prompt, show_progress, **kwargs)

        return list(zip(ids, responses))

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary statistics.

        Returns:
            Dictionary with cost and token statistics
        """
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.total_tokens,  # Approximation
            "completion_tokens": 0,  # Would need to track separately
            "model": self.model_name,
        }

    def reset_cost_tracking(self) -> None:
        """Reset cost tracking counters."""
        self.total_cost = 0.0
        self.total_tokens = 0
        logger.info("Reset cost tracking")
