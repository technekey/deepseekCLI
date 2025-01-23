#!/usr/bin/env python3

import os
import sys
import argparse
import requests
import time
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin
import asyncio
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.application import run_in_terminal
from colorama import init as colorama_init, Fore, Style
from pygments import highlight
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.formatters import TerminalFormatter
import re

# Initialize colorama
colorama_init(autoreset=True)

# Default configuration
DEFAULT_CONFIG = {
    "api_base_url": "https://api.deepseek.com",
    "chat_completions_endpoint": "/chat/completions",
    "usage_endpoint": "/user/balance",
    "models_endpoint": "/models",
    "model": "deepseek-chat",
    "temperature": 0.2,
    "max_tokens": 8192,
    "system_prompt": "You are a highly knowledgeable and accurate assistant. Please provide correct and concise answers to the user's questions."
}

# Define directories for sessions, logs, and config
DEEPSEEK_DIR = Path.home() / '.deepseek'
SESSIONS_DIR = DEEPSEEK_DIR / 'sessions'
LOGS_DIR = DEEPSEEK_DIR / 'logs'
DEFAULT_CONFIG_PATH = DEEPSEEK_DIR / 'config.json'

# Rate limit retry delay
RATE_LIMIT_RETRY_DELAY = 5  # seconds

def initialize_directories():
    """Create necessary directories if they do not exist."""
    try:
        DEEPSEEK_DIR.mkdir(parents=True, exist_ok=True)
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print_error(f"Failed to create necessary directories: {e}")
        sys.exit(1)

def setup_logging():
    """Configure logging to write to the logs directory with rotation."""
    log_file = LOGS_DIR / 'deepseek_cli.log'
    handler = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=5*1024*1024,  # 5 MB per log file
        backupCount=5,         # Keep up to 5 backup files
        encoding='utf-8'
    )
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

def get_api_key():
    """Retrieve the API key from the environment variable."""
    api_key = os.getenv('DEEPSEEK_API')
    if not api_key:
        print_error("DEEPSEEK_API environment variable is not set.")
        sys.exit(1)
    return api_key

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Deepseek AI Command-Line Interface Tool",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  Send a message:
    python deepseek.py send "Hello, how are you?"

  Start a new interactive session:
    python deepseek.py interactive

  Continue an existing interactive session named 'session1':
    python deepseek.py interactive --attach session1

  Attach 'session1' to a send command for context:
    python deepseek.py send "Tell me more about that." --attach session1

  List available models:
    python deepseek.py list-models

  Get API usage information:
    python deepseek.py usage

  Reset all conversation histories:
    python deepseek.py reset

  View conversation histories:
    python deepseek.py history

  List all sessions:
    python deepseek.py list-sessions
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Common arguments
    parser.add_argument(
        '--config',
        type=str,
        help='Path to the configuration file (default: ~/.deepseek/config.json)',
        default=str(DEFAULT_CONFIG_PATH)
    )

    # Subparser for sending a message
    send_parser = subparsers.add_parser(
        'send',
        help='Send a message to Deepseek AI',
        description='Send a single message to Deepseek AI and receive a response. Optionally attach to an existing session for context.'
    )
    send_parser.add_argument(
        'message',
        type=str,
        help='The message to send to Deepseek AI.'
    )
    send_parser.add_argument(
        '--model',
        type=str,
        help='Specify the model to use. Overrides config file.',
    )
    send_parser.add_argument(
        '--stream',
        action='store_true',
        help='Enable streaming responses from Deepseek AI.'
    )
    send_parser.add_argument(
        '--temperature',
        type=float,
        help='Sampling temperature for response generation. Higher values make output more random. Overrides config file.'
    )
    send_parser.add_argument(
        '--max_tokens',
        type=int,
        help='Maximum number of tokens in the response. Overrides config file.'
    )
    send_parser.add_argument(
        '--attach',
        type=str,
        help='Attach to an existing session by name or number for context.',
        default=None
    )
    send_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Display the full JSON response from Deepseek AI for detailed debugging.'
    )
    send_parser.add_argument(
        '--beautify',
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        nargs='?',
        const=True,
        default=True,
        help='Enable colorized output for better readability (default: true). Set to false to disable by passing "false", "0", or "no".'
    )

    # Subparser for resetting conversation histories
    reset_parser = subparsers.add_parser(
        'reset',
        help='Reset all conversation histories',
        description='Delete all saved conversation histories.'
    )

    # Subparser for viewing conversation histories
    history_parser = subparsers.add_parser(
        'history',
        help='View conversation histories',
        description='List and view saved conversation histories.'
    )
    history_parser.add_argument(
        '--beautify',
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        nargs='?',
        const=True,
        default=True,
        help='Enable colorized output for better readability when viewing history (default: true). Set to false to disable by passing "false", "0", or "no".'
    )

    # Subparser for listing all sessions
    list_sessions_parser = subparsers.add_parser(
        'list-sessions',
        help='List all conversation sessions',
        description='Display a list of all existing conversation sessions.'
    )

    # Subparser for usage info
    usage_parser = subparsers.add_parser(
        'usage',
        help='Get API usage information',
        description='Retrieve information about your API usage and account balance.'
    )

    # Subparser for listing models
    list_models_parser = subparsers.add_parser(
        'list-models',
        help='List available models',
        description='Fetch and display the list of available AI models from Deepseek.'
    )
    list_models_parser.add_argument(
        '--model',
        type=str,
        help='Specify a specific model to get details about. If omitted, lists all models.',
    )

    # Subparser for interactive mode
    interactive_parser = subparsers.add_parser(
        'interactive',
        help='Start an interactive conversation',
        description='Begin an interactive chat session with Deepseek AI. You can create a new session or continue an existing one.'
    )
    interactive_parser.add_argument(
        '--attach',
        type=str,
        help='Attach to an existing session by name or number. If not provided, a new session is initiated.',
        default=None
    )
    interactive_parser.add_argument(
        '--model',
        type=str,
        help='Specify the model to use. Overrides config file.',
    )
    interactive_parser.add_argument(
        '--stream',
        action='store_true',
        help='Enable streaming responses from Deepseek AI.'
    )
    interactive_parser.add_argument(
        '--temperature',
        type=float,
        help='Sampling temperature for response generation. Higher values make output more random. Overrides config file.'
    )
    interactive_parser.add_argument(
        '--max_tokens',
        type=int,
        help='Maximum number of tokens in the response. Overrides config file.'
    )
    interactive_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Display the full JSON response from Deepseek AI for detailed debugging.'
    )
    interactive_parser.add_argument(
        '--beautify',
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        nargs='?',
        const=True,
        default=True,
        help='Enable colorized output for better readability (default: true). Set to false to disable by passing "false", "0", or "no".'
    )

    return parser.parse_args()

def load_config(config_path):
    """Load configuration from the specified config file."""
    if not Path(config_path).exists():
        print_warning(f"Configuration file not found at {config_path}.")
        create_config = input(f"Do you want to create a default config file at {config_path}? (y/n): ").strip().lower()
        if create_config in ['y', 'yes']:
            try:
                with open(config_path, 'w') as f:
                    json.dump(DEFAULT_CONFIG, f, indent=4)
                print_success(f"Default configuration file created at {config_path}.")
                return DEFAULT_CONFIG
            except Exception as e:
                print_error(f"Failed to create config file: {e}")
                sys.exit(1)
        else:
            print_error("Cannot proceed without a configuration file. Exiting.")
            sys.exit(1)
    else:
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            # Validate config
            for key in DEFAULT_CONFIG:
                if key not in config:
                    print_warning(f"'{key}' not found in config. Using default value.")
                    config[key] = DEFAULT_CONFIG[key]
            return config
        except json.JSONDecodeError:
            print_error("Configuration file is not a valid JSON.")
            sys.exit(1)
        except Exception as e:
            print_error(f"Failed to read config file: {e}")
            sys.exit(1)

def merge_config_and_args(config, args):
    """Override config with command-line arguments if provided."""
    # Define a mapping of argument names to config keys
    arg_to_config = {
        'model': 'model',
        'temperature': 'temperature',
        'max_tokens': 'max_tokens'
    }

    for arg, config_key in arg_to_config.items():
        arg_value = getattr(args, arg, None)
        if arg_value is not None:
            config[config_key] = arg_value

    return config

def load_history(session_file):
    """Load conversation history from the session file."""
    if session_file.exists():
        try:
            with open(session_file, 'r') as f:
                history = json.load(f)
                if isinstance(history, list):
                    return history
                else:
                    print_warning("Session file is corrupted. Starting a new conversation.")
                    return []
        except json.JSONDecodeError:
            print_warning("Failed to decode session file. Starting a new conversation.")
            return []
    else:
        return []

def save_history(session_file, history):
    """Save conversation history to the session file."""
    try:
        with open(session_file, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print_error(f"Failed to save session history: {e}")
        logging.error(f"Failed to save session history: {e}")

def reset_history():
    """Reset (delete) all conversation histories."""
    if SESSIONS_DIR.exists():
        try:
            for session_file in SESSIONS_DIR.glob('*.json'):
                session_file.unlink()
            print_success("All conversation histories have been reset.")
            logging.info("All conversation histories reset by user.")
        except Exception as e:
            print_error(f"Failed to reset histories: {e}")
            logging.error(f"Failed to reset histories: {e}")
    else:
        print_warning("No conversation histories to reset.")

def view_history(beautify=False):
    """Display the conversation histories."""
    if not SESSIONS_DIR.exists():
        print_warning("No conversation histories found.")
        return

    sessions = sorted(SESSIONS_DIR.glob('*.json'))
    if not sessions:
        print_warning("No conversation histories found.")
        return

    print("Available Sessions:")
    for idx, session_file in enumerate(sessions, start=1):
        session_name = session_file.stem
        print(f"{idx}. {session_name}")

    try:
        choice = input("Enter the session number to view its history (or press Enter to return): ")
        if not choice.strip():
            return
        choice = int(choice)
        if 1 <= choice <= len(sessions):
            selected_session = sessions[choice - 1]
            history = load_history(selected_session)
            session_name = selected_session.stem
            print(f"\nConversation History for Session '{session_name}':")
            for message in history:
                role = message.get('role', 'unknown').capitalize()
                content = message.get('content', '')
                if beautify:
                    if role == 'Assistant':
                        print(f"{Fore.GREEN}{role}:{Style.RESET_ALL} {Fore.LIGHTBLACK_EX}{content}{Style.RESET_ALL}")
                    elif role == 'User':
                        print(f"{Fore.YELLOW}{role}:{Style.RESET_ALL} {content}")
                    else:
                        print(f"{role}: {content}")
                else:
                    print(f"{role}: {content}")
        else:
            print_error("Invalid session number.", beautify)
    except ValueError:
        print_error("Invalid input. Please enter a valid session number.", beautify)

def list_sessions():
    """List all existing conversation sessions."""
    if not SESSIONS_DIR.exists():
        print_warning("No conversation sessions found.")
        return

    sessions = sorted(SESSIONS_DIR.glob('*.json'))
    if not sessions:
        print_warning("No conversation sessions found.")
        return

    print("Existing Sessions:")
    for idx, session_file in enumerate(sessions, start=1):
        session_name = session_file.stem
        print(f"{idx}. {session_name}")

def handle_rate_limit(response):
    """Handle rate limiting based on the response."""
    if response.status_code == 429:
        retry_after = int(response.headers.get('Retry-After', RATE_LIMIT_RETRY_DELAY))
        print_error(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
        time.sleep(retry_after)
        return True
    return False

def resolve_session(session_identifier):
    """Resolve session identifier (name or number) to session file."""
    sessions = sorted(SESSIONS_DIR.glob('*.json'))
    if not sessions:
        print_warning("No sessions available.")
        return None

    # Check if session_identifier is a digit (number)
    if session_identifier.isdigit():
        index = int(session_identifier) - 1
        if 0 <= index < len(sessions):
            return sessions[index]
        else:
            print_error(f"Invalid session number: {session_identifier}")
            return None
    else:
        # Treat as session name
        session_file = SESSIONS_DIR / f"{session_identifier}.json"
        if session_file.exists():
            return session_file
        else:
            print_error(f"Session '{session_identifier}' does not exist.")
            return None

def fetch_available_models(api_key, config):
    """Fetch available models from the API."""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }
    url = urljoin(config['api_base_url'], config['models_endpoint'])

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            models_response = response.json()
            models = models_response.get('models') or models_response.get('data') or models_response
            if isinstance(models, list) and models:
                return [model.get('name') or model.get('id') or model.get('model') for model in models if isinstance(model, dict)]
            else:
                return []
        else:
            print_error(f"Failed to fetch models for interactive mode. Status Code: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print_error(f"Failed to fetch models for interactive mode: {e}")
        return []

def log_pretty(label, data):
    """Helper function to log pretty-printed JSON data."""
    try:
        pretty_data = json.dumps(data, indent=2)
        logging.info(f"{label}: {pretty_data}")
    except (TypeError, ValueError) as e:
        logging.error(f"Failed to pretty-print JSON for {label}: {e}")

def print_stats(stats, beautify=True):
    """Print statistics in the desired single-line format."""
    session_name = stats['session_name']
    model = stats['model']
    temperature = stats['temperature']
    total_tokens = stats.get('total_tokens', 'N/A')
    completion_tokens = stats.get('completion_tokens', 'N/A')
    prompt_tokens = stats.get('prompt_tokens', 'N/A')

    if all(isinstance(tok, int) for tok in [total_tokens, completion_tokens, prompt_tokens]):
        tokens_str = f"Tokens: Total/Completion/Prompt: {total_tokens}/{completion_tokens}/{prompt_tokens} |"
    else:
        tokens_str = "Tokens: Total/Completion/Prompt: N/A |"

    stats_line = f"{session_name} | {model} | {temperature} | {tokens_str}"

    if beautify:
        # Apply color, e.g., cyan
        stats_line_colored = f"{Fore.CYAN}{stats_line}{Style.RESET_ALL}"
        print(stats_line_colored)
    else:
        print(stats_line)

def highlight_code_blocks(text, beautify=True):
    """
    Detect Markdown code blocks in the text and apply syntax highlighting.
    Supports code blocks with language specification, e.g., ```bash.
    """
    if not beautify:
        return text  # Return plain text without highlighting

    # Regular expression to detect code blocks
    code_block_pattern = re.compile(
        r'```(?P<lang>\w+)?\n(?P<code>.*?)```',
        re.DOTALL
    )

    def replacer(match):
        lang = match.group('lang')
        code = match.group('code')
        if lang:
            try:
                lexer = get_lexer_by_name(lang, stripall=True)
            except Exception:
                lexer = TextLexer()
        else:
            lexer = TextLexer()
        formatter = TerminalFormatter()
        highlighted_code = highlight(code, lexer, formatter)
        return highlighted_code

    # Replace all code blocks with highlighted code
    highlighted_text = code_block_pattern.sub(replacer, text)
    return highlighted_text

def apply_markdown_formatting(text, beautify=True):
    """Apply markdown formatting for bold and headings."""
    if not beautify:
        return text

    # Apply headings
    # ### Heading3
    text = re.sub(r'^### (.+)', lambda m: f"{Fore.YELLOW}{Style.BRIGHT}{m.group(1)}{Style.RESET_ALL}", text, flags=re.MULTILINE)
    # ## Heading2
    text = re.sub(r'^## (.+)', lambda m: f"{Fore.MAGENTA}{Style.BRIGHT}{m.group(1)}{Style.RESET_ALL}", text, flags=re.MULTILINE)
    # # Heading1
    text = re.sub(r'^# (.+)', lambda m: f"{Fore.CYAN}{Style.BRIGHT}{m.group(1)}{Style.RESET_ALL}", text, flags=re.MULTILINE)

    # Apply bold
    text = re.sub(r'\*\*(.+?)\*\*', lambda m: f"{Style.BRIGHT}{m.group(1)}{Style.RESET_ALL}", text)

    return text

def process_text(text, beautify=True):
    """Process text by highlighting code blocks and applying markdown formatting."""
    if not beautify:
        return text
    # First, highlight code blocks
    text = highlight_code_blocks(text, beautify=True)
    # Then, apply markdown formatting
    text = apply_markdown_formatting(text, beautify=True)
    return text

def print_success(message, beautify=True):
    """Print success messages."""
    if beautify:
        print(f"{Fore.GREEN}{message}{Style.RESET_ALL}")
    else:
        print(message)

def print_warning(message, beautify=True):
    """Print warning messages."""
    if beautify:
        print(f"{Fore.YELLOW}{message}{Style.RESET_ALL}")
    else:
        print(message)

def print_error(message, beautify=True):
    """Print error messages."""
    if beautify:
        print(f"{Fore.RED}{message}{Style.RESET_ALL}")
    else:
        print(message)

def print_info(message, beautify=True):
    """Print informational messages."""
    if beautify:
        print(f"{Fore.BLUE}{message}{Style.RESET_ALL}")
    else:
        print(message)

def send_message(api_key, message, config, stream=False, session_file=None, verbose=False, beautify=True):
    """Send a message to Deepseek AI and return the response."""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    # If session_file is provided, load history
    if session_file:
        history = load_history(session_file)
        # Append the user's message to history
        history.append({"role": "user", "content": message})
        session_name = session_file.stem
    else:
        # For standalone send without history
        history = [
            {
                "role": "system",
                "content": config['system_prompt']  # Use system_prompt from config
            },
            {
                "role": "user",
                "content": message
            }
        ]
        session_name = "Standalone"

    # Construct the payload
    payload = {
        'model': config['model'],
        'messages': history,
        'stream': stream,
        'temperature': config['temperature'],
        'max_tokens': config['max_tokens']
    }

    url = urljoin(config['api_base_url'], config['chat_completions_endpoint'])

    # Debug: Print request details if verbose
    if verbose:
        print_info(f"Sending POST request to URL: {url}", beautify)
        print_info(f"Payload:\n{json.dumps(payload, indent=2)}", beautify)

    while True:
        try:
            response = requests.post(url, headers=headers, json=payload)
            # Debug: Print response details if verbose
            if verbose:
                print_info(f"Received response with status code: {response.status_code}", beautify)
                print_info(f"Response body: {response.text}", beautify)

            if response.status_code == 200:
                response_json = response.json()
                # Extract assistant's reply
                assistant_message = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                usage = response_json.get('usage', {})
                if assistant_message:
                    if verbose:
                        # Display full JSON
                        print_info(f"Full JSON Response:\n{json.dumps(response_json, indent=2)}", beautify)
                    else:
                        # Process text: highlight code blocks and apply markdown formatting
                        formatted_message = process_text(assistant_message, beautify)
                        # Prepare response header
                        response_header = "Deepseek AI Response:" if not beautify else f"{Fore.GREEN}Deepseek AI Response:{Style.RESET_ALL}"
                        # Display the formatted message
                        print(f"{response_header}\n{formatted_message}", flush=True)
                        # Print stats in single-line format
                        stats = {
                            'session_name': session_name,
                            'model': config['model'],
                            'temperature': config['temperature'],
                            'total_tokens': usage.get('total_tokens', 'N/A'),
                            'completion_tokens': usage.get('completion_tokens', 'N/A'),
                            'prompt_tokens': usage.get('prompt_tokens', 'N/A')
                        }
                        print_stats(stats, beautify)

                        # Log the request and response as pretty-printed JSON
                        log_pretty("Request", payload)
                        log_pretty("Response", response_json)

                        # Optionally append assistant's reply to history
                        if session_file:
                            history.append({"role": "assistant", "content": assistant_message})
                            # Save updated history
                            save_history(session_file, history)
                else:
                    print_warning("Deepseek AI did not return a message.", beautify)
                return response_json
            elif response.status_code == 429:
                if handle_rate_limit(response):
                    continue
            elif 400 <= response.status_code < 500:
                print_error(f"Client Error {response.status_code}: {response.text}", beautify)
                logging.error(f"Client Error {response.status_code}: {response.text}")
                return None
            elif 500 <= response.status_code < 600:
                print_error(f"Server Error {response.status_code}: {response.text}", beautify)
                print_warning(f"Retrying in {RATE_LIMIT_RETRY_DELAY} seconds...", beautify)
                logging.error(f"Server Error {response.status_code}: {response.text}")
                time.sleep(RATE_LIMIT_RETRY_DELAY)
                continue
            else:
                print_error(f"Unexpected status code {response.status_code}: {response.text}", beautify)
                logging.error(f"Unexpected status code {response.status_code}: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print_error(f"Request failed: {e}", beautify)
            logging.error(f"Request failed: {e}")
            print_warning(f"Retrying in {RATE_LIMIT_RETRY_DELAY} seconds...", beautify)
            time.sleep(RATE_LIMIT_RETRY_DELAY)

def send_message_interactive(api_key, user_input, session_file, config, stream=False, verbose=False, beautify=True, stats=None):
    """Send a message within an interactive session and update history."""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    # Load existing conversation history
    history = load_history(session_file)

    # Append the user's message to history
    history.append({"role": "user", "content": user_input})

    # Construct the payload with history
    payload = {
        'model': config['model'],
        'messages': history,
        'stream': stream,
        'temperature': config['temperature'],
        'max_tokens': config['max_tokens']
    }

    url = urljoin(config['api_base_url'], config['chat_completions_endpoint'])

    # Debug: Print request details if verbose
    if verbose:
        print_info(f"Sending POST request to URL: {url}", beautify)
        print_info(f"Payload:\n{json.dumps(payload, indent=2)}", beautify)

    while True:
        try:
            response = requests.post(url, headers=headers, json=payload)
            # Debug: Print response details if verbose
            if verbose:
                print_info(f"Received response with status code: {response.status_code}", beautify)
                print_info(f"Response body: {response.text}", beautify)

            if response.status_code == 200:
                response_json = response.json()
                # Extract assistant's reply
                assistant_message = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                usage = response_json.get('usage', {})
                if assistant_message:
                    if verbose:
                        # Display full JSON
                        print_info(f"Full JSON Response:\n{json.dumps(response_json, indent=2)}", beautify)
                    else:
                        # Process text: highlight code blocks and apply markdown formatting
                        formatted_message = process_text(assistant_message, beautify)
                        # Prepare response header
                        response_header = "Deepseek AI Response:" if not beautify else f"{Fore.GREEN}Deepseek AI Response:{Style.RESET_ALL}"
                        # Display the formatted message
                        print(f"{response_header}\n{formatted_message}", flush=True)
                        # Update and print stats in single-line format
                        stats['total_tokens'] = usage.get('total_tokens', 'N/A')
                        stats['completion_tokens'] = usage.get('completion_tokens', 'N/A')
                        stats['prompt_tokens'] = usage.get('prompt_tokens', 'N/A')
                        print_stats(stats, beautify)

                        # Log the request and response as pretty-printed JSON
                        log_pretty("Request", payload)
                        log_pretty("Response", response_json)

                        # Optionally append assistant's reply to history
                        if session_file:
                            history.append({"role": "assistant", "content": assistant_message})
                            # Save updated history
                            save_history(session_file, history)
                else:
                    print_warning("Deepseek AI did not return a message.", beautify)
                return response_json
            elif response.status_code == 429:
                if handle_rate_limit(response):
                    continue
            elif 400 <= response.status_code < 500:
                print_error(f"Client Error {response.status_code}: {response.text}", beautify)
                logging.error(f"Client Error {response.status_code}: {response.text}")
                return None
            elif 500 <= response.status_code < 600:
                print_error(f"Server Error {response.status_code}: {response.text}", beautify)
                print_warning(f"Retrying in {RATE_LIMIT_RETRY_DELAY} seconds...", beautify)
                logging.error(f"Server Error {response.status_code}: {response.text}")
                time.sleep(RATE_LIMIT_RETRY_DELAY)
                continue
            else:
                print_error(f"Unexpected status code {response.status_code}: {response.text}", beautify)
                logging.error(f"Unexpected status code {response.status_code}: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print_error(f"Request failed: {e}", beautify)
            logging.error(f"Request failed: {e}")
            print_warning(f"Retrying in {RATE_LIMIT_RETRY_DELAY} seconds...", beautify)
            time.sleep(RATE_LIMIT_RETRY_DELAY)

def run_change_temperature(config, stats):
    """Prompt user to change temperature."""
    try:
        new_temp_str = input(f"{Fore.CYAN}Enter new temperature (current: {stats['temperature']}): {Style.RESET_ALL}")
        new_temp = float(new_temp_str)
        if 0 <= new_temp <= 2:
            config['temperature'] = new_temp
            stats['temperature'] = new_temp
            print_success(f"Temperature updated to {new_temp}.")
        else:
            print_error("Temperature must be between 0 and 2.")
    except ValueError:
        print_error("Invalid temperature value.")

def run_change_model(api_key, config, stats, available_models):
    """Prompt user to change model."""
    if not available_models:
        print_error("No available models to select.")
        return
    print_info("Available Models:", beautify=True)
    for idx, model in enumerate(available_models, start=1):
        print_info(f"{idx}. {model}", beautify=True)
    try:
        choice_str = input(f"{Fore.CYAN}Select a model by number (current: {stats['model']}): {Style.RESET_ALL}")
        choice = int(choice_str)
        if 1 <= choice <= len(available_models):
            selected_model = available_models[choice - 1]
            config['model'] = selected_model
            stats['model'] = selected_model
            print_success(f"Model updated to {selected_model}.")
        else:
            print_error("Invalid model number.")
    except ValueError:
        print_error("Invalid input. Please enter a number.")

async def interactive_mode_async(api_key, attach=None, config=None, stream=False, verbose=False, beautify=True):
    """Start an interactive conversation with Deepseek AI asynchronously."""
    session_file = None
    session_name = None

    if attach:
        session_file = resolve_session(attach)
        if session_file:
            session_name = session_file.stem
            print_success(f"Continuing conversation in session '{session_name}'.", beautify)
        else:
            print_warning("Starting a new conversation.", beautify)
    else:
        print_warning("Starting a new conversation.", beautify)

    if not session_file:
        # Create a new session with a timestamp
        session_name = f"session_{int(time.time())}"
        session_file = SESSIONS_DIR / f"{session_name}.json"
        # Initialize with system prompt from config
        history = [
            {
                "role": "system",
                "content": config['system_prompt']  # Use system_prompt from config
            }
        ]
        save_history(session_file, history)
        print_success(f"New session created: '{session_name}'", beautify)

    # Initialize stats
    stats = {
        'session_name': session_name,
        'model': config['model'],
        'temperature': config['temperature'],
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0
    }

    print_info("Interactive mode. Type 'exit' or 'quit' to end the session.\n", beautify)

    # Fetch available models for dynamic selection
    available_models = fetch_available_models(api_key, config)
    if not available_models:
        available_models = [config['model']]  # Fallback to current model

    # Define a callable for the bottom toolbar using HTML for styling
    def get_bottom_toolbar():
        if beautify:
            return HTML(
                f"<cyan>Press Ctrl+D to send the message, Ctrl+C to cancel, Ctrl+T to change temperature, Ctrl+X to change model.</cyan>"
            )
        else:
            return "Press Ctrl+D to send the message, Ctrl+C to cancel, Ctrl+T to change temperature, Ctrl+X to change model."

    # Initialize PromptSession from prompt_toolkit with custom key bindings
    bindings = KeyBindings()

    @bindings.add('c-d')
    def _(event):
        """
        Handle Ctrl+D to send the message instead of exiting.
        """
        buffer = event.app.current_buffer
        buffer.validate_and_handle()

    @bindings.add('c-t')
    def _(event):
        """
        Handle Ctrl+T to change temperature.
        """
        run_in_terminal(lambda: run_change_temperature(config, stats))

    @bindings.add('c-x')  # Changed to 'c-x' (Ctrl+X)
    def _(event):
        """
        Handle Ctrl+X to change model.
        """
        run_in_terminal(lambda: run_change_model(api_key, config, stats, available_models))

    session = PromptSession(
        history=InMemoryHistory(),
        auto_suggest=AutoSuggestFromHistory(),
        key_bindings=bindings
    )

    while True:
        try:
            user_input = await session.prompt_async(
                "You: ",
                multiline=True,
                bottom_toolbar=get_bottom_toolbar,
                complete_style=CompleteStyle.READLINE_LIKE
            )
            if user_input.strip().lower() in ['exit', 'quit']:
                print_success("Ending interactive session.", beautify)
                break
            if not user_input.strip():
                continue  # Skip empty inputs

            send_message_interactive(
                api_key=api_key,
                user_input=user_input,
                session_file=session_file,
                config=config,
                stream=stream,
                verbose=verbose,
                beautify=beautify,
                stats=stats  # Pass the stats dictionary to update tokens
            )
        except (KeyboardInterrupt, EOFError):
            print_success("\nEnding interactive session.", beautify)
            break

def interactive_mode(api_key, attach=None, config=None, stream=False, verbose=False, beautify=True):
    """Start an interactive conversation with Deepseek AI asynchronously."""
    try:
        asyncio.run(
            interactive_mode_async(
                api_key=api_key,
                attach=attach,
                config=config,
                stream=stream,
                verbose=verbose,
                beautify=beautify
            )
        )
    except RuntimeError as e:
        print_error(f"Runtime Error: {e}", beautify)
        print_warning("If you're using an interactive environment like Jupyter, please run the script in a standard terminal.", beautify)

def get_usage(api_key, config):
    """Retrieve API usage information."""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }
    url = urljoin(config['api_base_url'], config['usage_endpoint'])

    print_info(f"Sending GET request to URL: {url}", beautify=True)

    try:
        response = requests.get(url, headers=headers)
        print_info(f"Usage Status: {response.status_code}", beautify=True)
        if response.status_code == 200:
            usage_info = response.json()
            print("API Usage Information:")
            print(json.dumps(usage_info, indent=2))
            logging.info(f"Usage Info: {usage_info}")
        else:
            print_error(f"Failed to retrieve usage info. Status Code: {response.status_code}", beautify=True)
            print_error(f"Response: {response.text}", beautify=True)
            logging.error(f"Failed to retrieve usage info. Status Code: {response.status_code}, Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print_error(f"Usage info request failed: {e}", beautify=True)
        logging.error(f"Usage info request failed: {e}")

def list_models(api_key, config, specific_model=None):
    """List available models."""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }
    url = urljoin(config['api_base_url'], config['models_endpoint'])

    print_info(f"Sending GET request to URL: {url}", beautify=True)

    try:
        response = requests.get(url, headers=headers)
        print_info(f"List Models Status: {response.status_code}", beautify=True)
        if response.status_code == 200:
            models_response = response.json()
            # Adjust based on actual API response structure
            models = models_response.get('models') or models_response.get('data') or models_response
            if isinstance(models, list) and models:
                if specific_model:
                    # Check if the specific model exists
                    matched_models = [model_item for model_item in models if
                                      model_item.get('name') == specific_model or
                                      model_item.get('id') == specific_model or
                                      model_item.get('model') == specific_model]
                    if matched_models:
                        print(f"Details for Model '{specific_model}':")
                        print(json.dumps(matched_models[0], indent=2))
                    else:
                        print_warning(f"Model '{specific_model}' not found.", beautify=True)
                else:
                    print("Available Models:")
                    for model_item in models:
                        if isinstance(model_item, dict):
                            # Adjust the key based on actual model structure
                            model_name = model_item.get('name') or model_item.get('id') or model_item.get('model') or str(model_item)
                            print(f"- {model_name}")
                        else:
                            print(f"- {model_item}")
            else:
                print_warning("No models found.", beautify=True)
            logging.info(f"Models: {models}")
        else:
            print_error(f"Failed to retrieve models. Status Code: {response.status_code}", beautify=True)
            print_error(f"Response: {response.text}", beautify=True)
            logging.error(f"Failed to retrieve models. Status Code: {response.status_code}, Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print_error(f"List models request failed: {e}", beautify=True)
        logging.error(f"List models request failed: {e}")

def main():
    """Main function to handle command-line interactions."""
    initialize_directories()
    setup_logging()
    args = parse_arguments()

    # Load configuration
    config_path = Path(args.config).expanduser()
    config = load_config(config_path)

    # Merge command-line arguments with config
    config = merge_config_and_args(config, args)

    api_key = get_api_key()

    if args.command == 'send':
        session_file = None
        session_name = "Standalone"
        if args.attach:
            session_file = resolve_session(args.attach)
            if session_file:
                session_name = session_file.stem
                print_success(f"Attaching to session '{session_name}' for context.", args.beautify)
            else:
                print_warning("Proceeding without attaching to a session.", args.beautify)
        response = send_message(
            api_key=api_key,
            message=args.message,
            config=config,
            stream=args.stream,
            session_file=session_file,
            verbose=args.verbose,
            beautify=args.beautify
        )
        if not response:
            print_error("Failed to get a response from Deepseek AI.", args.beautify)

    elif args.command == 'reset':
        reset_history()

    elif args.command == 'history':
        view_history(beautify=args.beautify)

    elif args.command == 'list-sessions':
        list_sessions()

    elif args.command == 'usage':
        get_usage(api_key, config)

    elif args.command == 'list-models':
        specific_model = args.model
        list_models(api_key, config, specific_model)

    elif args.command == 'interactive':
        interactive_mode(
            api_key=api_key,
            attach=args.attach,
            config=config,
            stream=args.stream,
            verbose=args.verbose,
            beautify=args.beautify
        )

    else:
        print_warning("No command provided. Use -h for help.", beautify=True)

if __name__ == '__main__':
    main()
