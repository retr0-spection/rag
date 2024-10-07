from typing import List, Dict, Optional, Union, Any
import requests
from bs4 import BeautifulSoup
from typing import Dict, Union, Annotated
from urllib.parse import urlencode
from langchain_core.tools import tool
from app.ingestion.utils import Ingestion
import subprocess
import uuid
import os
import docker
import traceback
from docker.errors import ContainerError, ImageNotFound, APIError




@tool
def web_search_and_extract(query: str, num_links: int = 3) -> List[Dict[str, str]]:
    """
    Perform a web search and extract content from the top results.

    This function combines the search_for_links and extract_content functions to provide
    a comprehensive web search and content extraction tool. It can be used as a custom tool
    in LangChain or a node in LangGraph for tasks requiring web-based information gathering.

    Args:
        query (str): The search query string.
        num_links (int, optional): The number of top search results to process. Defaults to 3.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing 'title', 'url', and 'content'
                              of a search result. If errors occur, they will be included in the respective
                              dictionary with an 'error' key.

    Example usage in LangChain:
        tools = [Tool(name="Web Search and Extract", func=web_search_and_extract,
                      description="Searches the web and extracts content from top results")]

    Example usage in LangGraph:
        search_extract_node = FunctionNode(web_search_and_extract, name="web_search_and_extract")
    """
    # Search for links
    links = search_for_links(query, num_links)
    results = []

    # Extract content from each link
    for link in links:
        if isinstance(link, dict) and 'url' in link:
            content = extract_content(link['url'])
            results.append(content)
        else:
            results.append({'error': "Invalid link format from search results"})

    return results

def search_for_links(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Perform a Google search and extract relevant links.

    This function can be used as a tool in LangChain or a node in LangGraph for web search tasks.
    It simulates a Google search and extracts the top results, returning their titles and URLs.

    Args:
        query (str): The search query string.
        num_results (int, optional): The number of search results to return. Defaults to 5.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing 'title' and 'url' of a search result.
                              Returns an error message string if an exception occurs.

    Example usage in LangChain:
        tools = [Tool(name="Web Search", func=search_for_links, description="Searches Google and returns relevant links")]

    Example usage in LangGraph:
        search_node = FunctionNode(search_for_links, name="web_search")
    """
    search_url = f"https://www.google.com/search?{urlencode({'q': query})}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # Fetch the search results page
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract search result links
        search_results = soup.find_all('div', class_='yuRUbf')
        links = []

        for result in search_results[:num_results]:
            link = result.find('a')['href']
            title = result.find('h3', class_='r').text if result.find('h3', class_='r') else "No title"
            links.append({'title': title, 'url': link})

        return links

    except requests.RequestException as e:
        return f"An error occurred during the web search: {e}"




def extract_content(url: str) -> Dict[str, str]:
    """
    Extract content from a given webpage URL.

    This function can be used as a tool in LangChain or a node in LangGraph for web scraping tasks.
    It visits the provided URL, extracts the title and main content of the webpage.

    Args:
        url (str): The URL of the webpage to extract content from.

    Returns:
        Dict[str, str]: A dictionary containing 'title', 'url', and 'content' of the webpage.
                        Returns a dictionary with an 'error' key if an exception occurs.

    Example usage in LangChain:
        tools = [Tool(name="Web Content Extractor", func=extract_content, description="Extracts content from a given URL")]

    Example usage in LangGraph:
        extract_node = FunctionNode(extract_content, name="content_extractor")
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # Fetch the webpage content
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the title
        title = soup.title.string if soup.title else "No title found"

        # Try to find the main content
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')

        if main_content:
            # Extract text from paragraphs in the main content
            paragraphs = main_content.find_all('p')
            content = "\n".join(p.get_text().strip() for p in paragraphs)
        else:
            # If no main content found, extract all paragraph text
            paragraphs = soup.find_all('p')
            content = "\n".join(p.get_text().strip() for p in paragraphs)

        # Truncate content if it's too long
        max_length = 5000  # Adjust as needed
        if len(content) > max_length:
            content = content[:max_length] + "..."

        return {
            'title': title,
            'url': url,
            'content': content
        }

    except requests.RequestException as e:
        return {'error': f"An error occurred while extracting content: {e}"}

@tool
def get_document_contents(document_name:Annotated[str, "Document name as seen in knowledge base"], user_id:Annotated[str, "user's id"]):
    """
    Retrieve/access document contents from knowledge base. Aurora can call this

    Args:
        document_name (str): Document name as seen in knowledge base.
        user_id (str): The ID of the user who owns the document.

    Returns:
        List[Dict]: A list of dictionaries, each containing a chunk's content and metadata. List is empty for empty files.
    """
    ingestion = Ingestion()

    results = ingestion.get_document_chunks(document_name, user_id)
    return results if len(results) else "no contents or file found."


@tool
async def run_python_code_in_container(code: str, timeout: int = 120, packages: str = '') -> Dict[str, Any]:
    """
    Executes a piece of Python code in a Docker container.

    Parameters:
    - code: The Python code to execute as a string.
    - timeout: Time in seconds to wait for the execution to finish. Default is 120 seconds.
    - packages: A string of space-separated package names to install using pip.

    Returns:
    A dictionary containing:
    - status: 'success' or 'error'
    - output: The output of the code execution or error message
    - error: Detailed error message if applicable
    - code: The code that was executed
    """
    client = docker.from_env()
    response = {
        "status": "error",
        "output": None,
        "error": None,
        "code": code
    }

    # Build the command to install packages if any are specified
    install_command = f"pip install {packages} && " if packages else ""

    try:
        # Spin up the container
        container = client.containers.run(
            image="python:3.10-slim",
            command=f"sh -c '{install_command} python -c \"{code.replace('\"', '\\\"')}\"'",
            detach=True,
            stdout=True,
            stderr=True
        )

        # Wait for the container to finish executing or time out
        result = container.wait(timeout=timeout)
        logs = container.logs()

        # If the container completed without issues
        if result['StatusCode'] == 0:
            response["status"] = "success"
            response["output"] = logs.decode('utf-8').strip()
        else:
            response["error"] = f"Non-zero exit code: {result['StatusCode']}"
            response["output"] = logs.decode('utf-8').strip()

    except ContainerError as e:
        response["error"] = f"Container error: {str(e)}"
        if hasattr(e, 'stderr'):
            response["output"] = e.stderr.decode('utf-8').strip() if e.stderr else "No error output available."
        elif hasattr(e, 'stdout'):
            response["output"] = e.stdout.decode('utf-8').strip() if e.stdout else "No standard output available."

    except ImageNotFound as e:
        response["error"] = "Image not found: " + str(e)

    except APIError as e:
        response["error"] = "Docker API error: " + str(e)

    except Exception as e:
        # Catch any other unforeseen errors
        response["error"] = "Unexpected error: " + str(e)

    finally:
        # Ensure the container is stopped and removed
        try:
            container.stop()
            container.remove()
        except Exception as cleanup_error:
            response["error"] = f"Cleanup error: {str(cleanup_error)}"

        client.close()

    return response


# async def fetch_customer_context(query_str: Annotated[str, "User prompt"], user_id: Annotated[str, "user id"]) -> str:
#     """Fetch internal resources and user documents that might be relevant to the query.

#     Args:
#         query_str (str): Query string with relevant keywords from user's query.
#         user_id (str): The ID of the user.

#     Returns:
#         str: String containing information that might be relevant to the query.
#     """
#     ingestion = Ingestion()
#     ingestion.get_or_create_collection('embeddings')


#     # First, try to find file matches
#     # file_matches = ingestion.query_file_names(query_str, user_id)

#     # Then, try to find tag matches
#     tag_matches = await ingestion.query_by_tags(query_str, user_id)

#     # If we have strong file matches, prioritize those
#     # if len(file_matches) and file_matches[0]['similarity'] > FILE_MATCH_THRESHOLD:
#     #     return "\n".join([match['text'] for match in file_matches])

#     # If we have strong tag matches, include those
#     tag_matched_files = [match for match in tag_matches if match['combined_score'] > TAG_MATCH_THRESHOLD]

#     # Perform semantic search
#     semantic_results = await ingestion.query(query_str, user_id, relevance_threshold=RELEVANCE_THRESHOLD)

#     # Combine and rank results
#     combined_results = rank_and_combine_results(tag_matched_files, semantic_results)

#     # Extract and combine relevant text
#     context_text = []
#     for result in combined_results:
#         if 'text' in result:
#             context_text.append(result['text'])
#         elif 'sample_text' in result:
#             context_text.append(result['sample_text'])

#     return "\n".join(context_text)
