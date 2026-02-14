"""
Datasets Loading Module

This module provides functionality for loading pre-defined datasets.

Functions:
- list_datasets(): List all available datasets
- load_data(): Load a specific dataset

Available Datasets:
- trump_tweet_archive: Trump Tweet Archives data
- state_of_the_union: State of the Union Addresses by US Presidents (1790-2018)
"""

import json
from ast import literal_eval
from io import StringIO
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
import requests
from loguru import logger


# =============================================================================
# Dataset Metadata
# =============================================================================
DATASETS_INFO = """
{
    "trump_tweet_archive": 
    {
        "description": "Tweets from the Trump Tweet Archives (https://www.thetrumparchive.com/)",
        "language": "english",
        "srl_model": "allennlp v0.9 -- srl-model-2018.05.25.tar.gz",
        "links": 
        {
            "raw": "https://www.dropbox.com/s/lxqz454n29iqktn/trump_archive.csv?dl=1",
            "sentences": "https://www.dropbox.com/s/coh4ergyrjeolen/split_sentences.json?dl=1",
            "srl_res": "https://www.dropbox.com/s/54lloy84ka8mycp/srl_res.json?dl=1"
        }
    },
    "state_of_the_union": 
    {
        "description": "State of the Union Addresses by US Presidents (1790-2018), sourced from the American Presidency Project",
        "language": "english",
        "srl_model": "",
        "links": 
        {
            "raw": "https://raw.githubusercontent.com/BrianWeinstein/state-of-the-union/master/transcripts.csv"
        }
    }
}
"""


def _get_datasets_dict() -> Dict[str, Any]:
    """
    Get dataset information dictionary
    
    Returns:
        Dict[str, Any]: Dataset metadata dictionary
    """
    return json.loads(DATASETS_INFO)


def list_datasets(verbose: bool = True) -> List[str]:
    """
    List all available datasets
    
    Args:
        verbose: Whether to print detailed information (default True)
    
    Returns:
        List[str]: List of dataset names
    
    Example:
        >>> list_datasets()
        Available Datasets:
        ===================
        1. trump_tweet_archive
           - Description: Tweets from the Trump Tweet Archives
           - Language: english
           - Available content: raw, sentences, srl_res
        ...
    """
    logger.debug("Listing available datasets")
    datasets = _get_datasets_dict()
    dataset_names = list(datasets.keys())
    
    if verbose:
        print("Available Datasets:")
        print("=" * 50)
        for i, name in enumerate(dataset_names, 1):
            info = datasets[name]
            print(f"\n{i}. {name}")
            print(f"   - Description: {info['description']}")
            print(f"   - Language: {info['language']}")
            print(f"   - Available content: {', '.join(info['links'].keys())}")
        print()
    
    logger.info(f"Found {len(dataset_names)} available datasets")
    return dataset_names


def get_dataset_info(dataset: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information for a specific dataset
    
    Args:
        dataset: Dataset name
    
    Returns:
        Dict[str, Any]: Dataset information, or None if not found
    
    Example:
        >>> info = get_dataset_info("trump_tweet_archive")
        >>> print(info['description'])
        Tweets from the Trump Tweet Archives
    """
    logger.debug(f"Getting dataset info for: {dataset}")
    datasets = _get_datasets_dict()
    info = datasets.get(dataset)
    if info:
        logger.info(f"Dataset info retrieved for: {dataset}")
    else:
        logger.warning(f"Dataset not found: {dataset}")
    return info


def load_data(
    dataset: str,
    content: Literal["raw", "sentences", "srl_res"] = "raw",
    verbose: bool = True,
) -> Union[pd.DataFrame, List[Any]]:
    """
    Load a specific dataset
    
    Args:
        dataset: Dataset name, options:
            - "trump_tweet_archive": Trump tweet data
            - "state_of_the_union": US Presidential State of the Union Addresses (1790-2018)
        content: Content type, options:
            - "raw": Raw CSV data (returns DataFrame)
            - "sentences": Sentence-segmented data (returns List)
            - "srl_res": SRL parsing results (returns List)
        verbose: Whether to print loading information (default True)
    
    Returns:
        Union[pd.DataFrame, List]: 
            - If content="raw", returns pandas DataFrame
            - Otherwise returns List
    
    Raises:
        ValueError: If dataset or content type doesn't exist
    
    Example:
        >>> # Load raw tweet data
        >>> df = load_data("trump_tweet_archive", content="raw")
        >>> print(f"Loaded {len(df)} tweets")
        
        >>> # Load SRL parsing results
        >>> srl_results = load_data("trump_tweet_archive", content="srl_res")
    """
    datasets = _get_datasets_dict()
    
    # Validate dataset name
    if dataset not in datasets:
        available = list(datasets.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset}'. "
            f"Available datasets: {available}"
        )
    
    dataset_info = datasets[dataset]
    
    # Validate content type
    if content not in dataset_info["links"]:
        available_content = list(dataset_info["links"].keys())
        raise ValueError(
            f"Content '{content}' not available for dataset '{dataset}'. "
            f"Available content types: {available_content}"
        )
    
    url = dataset_info["links"][content]
    
    logger.info(f"Loading dataset: {dataset}/{content} from {url}")
    if verbose:
        logger.info(f"Loading {dataset}/{content} from remote...")
    
    # Choose parsing method based on content type
    if content == "raw":
        # CSV data -> DataFrame
        logger.debug(f"Fetching CSV data from URL")
        response = requests.get(url)
        response.encoding = "utf8"
        result = pd.read_csv(StringIO(response.text))
        
        logger.info(f"Successfully loaded DataFrame with {len(result)} rows")
        if verbose:
            logger.info(f"Loaded DataFrame with {len(result)} rows")
    else:
        # JSON data -> List
        logger.debug(f"Fetching JSON data from URL")
        response = requests.get(url)
        result = literal_eval(response.text)
        
        logger.info(f"Successfully loaded {len(result)} items")
        if verbose:
            logger.info(f"Loaded {len(result)} items")
    
    return result


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "list_datasets",
    "get_dataset_info",
    "load_data",
    "DATASETS_INFO",
]

