#!/usr/bin/env python3

import os
import sys
import argparse
import requests
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

# Configuration: Update these based on actual API documentation
API_BASE_URL = "https://api.deepseek.com"
CHAT_COMPLETIONS_ENDPOINT = "/chat/completions"
USAGE_ENDPOINT = "/user/balance"  # Updated usage endpoint
MODELS_ENDPOINT = "/models"        # Endpoint to list models
RATE_LIMIT_RETRY_DELAY = 5        # Seconds to wait before retrying after rate limit

# Define directories for sessions and logs
DEEPSEEK_DIR = Path.home() / '.deepseek'
SESSIONS_DIR = DEEPSEEK_DIR / 'sessions'
LOGS_DIR = DEEPSEEK_DIR / 'logs'

# ANSI escape codes for colors (temporarily removed for troubleshooting)
# GRAY = '\033[90m'
# RESET = '\033[0m'

def initialize_directories():
    """Create necessary directories if they do not exist."""
    try:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Failed to create necessary directories. {e}")
        sys.exit(1)

def setup_logging():
    """Configure logging to write to the logs directory."""
    log_file = LOGS_DIR / 'deepseek_cli.log'
    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s'
    )

def get_api_key():
    """Retrieve the API key from the environment variable."""
    api_key = os.getenv('DEEPSEEK_API')
    if not api_key:
        print("Error: DEEPSEEK_API environment variable is not set.")
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

  Check API usage:
    python deepseek.py usage

  Reset all conversation histories:
    python deepseek.py reset

  View conversation histories:
    python deepseek.py history

  List all sessions:
    python deepseek.py list-sessions

  Perform a health check on the API:
    python deepseek.py health-check
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

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
        help='Specify the model to use. Defaults to "deepseek-chat".',
        default='deepseek-chat'
    )
    send_parser.add_argument(
        '--stream',
        action='store_true',
        help='Enable streaming responses from Deepseek AI.'
    )
    send_parser.add_argument(
        '--temperature',
        type=float,
        help='Sampling temperature for response generation. Higher values make output more random. Defaults to 0.2.',
        default=0.2
    )
    send_parser.add_argument(
        '--max_tokens',
        type=int,
        help='Maximum number of tokens in the response. Defaults to 150.',
        default=8192
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
        help='Display the full JSON response from Deepseek AI.'
    )
    send_parser.add_argument(
        '--beautify',
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=True,
        help='Enable colorized output (default: true). Set to false to disable.'
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

    # Subparser for listing all sessions
    list_sessions_parser = subparsers.add_parser(
        'list-sessions',
        help='List all conversation sessions',
        description='Display a list of all existing conversation sessions.'
    )

    # Subparser for health check
    health_parser = subparsers.add_parser(
        'health-check',
        help='Check API health status',
        description='Perform a health check to verify API connectivity and status.'
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
        help='Specify the model to use. Defaults to "deepseek-chat".',
        default='deepseek-chat'
    )
    interactive_parser.add_argument(
        '--stream',
        action='store_true',
        help='Enable streaming responses from Deepseek AI.'
    )
    interactive_parser.add_argument(
        '--temperature',
        type=float,
        help='Sampling temperature for response generation. Higher values make output more random. Defaults to 0.2.',
        default=0.2
    )
    interactive_parser.add_argument(
        '--max_tokens',
        type=int,
        help='Maximum number of tokens in the response. Defaults to 150.',
        default=8192
    )
    interactive_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Display the full JSON response from Deepseek AI.'
    )
    interactive_parser.add_argument(
        '--beautify',
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=True,
        help='Enable colorized output (default: true). Set to false to disable.'
    )

    return parser.parse_args()

def load_history(session_file):
    """Load conversation history from the session file."""
    if session_file.exists():
        try:
            with open(session_file, 'r') as f:
                history = json.load(f)
                if isinstance(history, list):
                    return history
                else:
                    print("Warning: Session file is corrupted. Starting a new conversation.")
                    return []
        except json.JSONDecodeError:
            print("Warning: Failed to decode session file. Starting a new conversation.")
            return []
    else:
        return []

def save_history(session_file, history):
    """Save conversation history to the session file."""
    try:
        with open(session_file, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Error: Failed to save session history. {e}")
        logging.error(f"Failed to save session history: {e}")

def reset_history():
    """Reset (delete) all conversation histories."""
    if SESSIONS_DIR.exists():
        try:
            for session_file in SESSIONS_DIR.glob('*.json'):
                session_file.unlink()
            print("All conversation histories have been reset.")
            logging.info("All conversation histories reset by user.")
        except Exception as e:
            print(f"Error: Failed to reset histories. {e}")
            logging.error(f"Failed to reset histories: {e}")
    else:
        print("No conversation histories to reset.")

def view_history():
    """Display the conversation histories."""
    if not SESSIONS_DIR.exists():
        print("No conversation histories found.")
        return

    sessions = sorted(SESSIONS_DIR.glob('*.json'))
    if not sessions:
        print("No conversation histories found.")
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
            print(f"\nConversation History for Session '{selected_session.stem}':")
            for message in history:
                role = message.get('role', 'unknown').capitalize()
                content = message.get('content', '')
                print(f"{role}: {content}")
        else:
            print("Invalid session number.")
    except ValueError:
        print("Invalid input. Please enter a valid session number.")

def list_sessions():
    """List all existing conversation sessions."""
    if not SESSIONS_DIR.exists():
        print("No conversation sessions found.")
        return

    sessions = sorted(SESSIONS_DIR.glob('*.json'))
    if not sessions:
        print("No conversation sessions found.")
        return

    print("Existing Sessions:")
    for idx, session_file in enumerate(sessions, start=1):
        session_name = session_file.stem
        print(f"{idx}. {session_name}")

def handle_rate_limit(response):
    """Handle rate limiting based on the response."""
    if response.status_code == 429:
        retry_after = int(response.headers.get('Retry-After', RATE_LIMIT_RETRY_DELAY))
        print(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
        time.sleep(retry_after)
        return True
    return False

def resolve_session(session_identifier):
    """Resolve session identifier (name or number) to session file."""
    sessions = sorted(SESSIONS_DIR.glob('*.json'))
    if not sessions:
        print("No sessions available.")
        return None

    # Check if session_identifier is a digit (number)
    if session_identifier.isdigit():
        index = int(session_identifier) - 1
        if 0 <= index < len(sessions):
            return sessions[index]
        else:
            print(f"Invalid session number: {session_identifier}")
            return None
    else:
        # Treat as session name
        session_file = SESSIONS_DIR / f"{session_identifier}.json"
        if session_file.exists():
            return session_file
        else:
            print(f"Session '{session_identifier}' does not exist.")
            return None

def send_message(api_key, message, model='deepseek-chat', stream=False, temperature=0.2, max_tokens=150, session_file=None, verbose=False, beautify=True):
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
    else:
        # For standalone send without history
        history = [
            {
                "role": "system",
                "content": "You are a highly knowledgeable and accurate assistant. Please provide correct and concise answers to the user's questions."
            },
            {
                "role": "user",
                "content": message
            }
        ]

    # Construct the payload
    payload = {
        'model': model,
        'messages': history,
        'stream': stream,
        'temperature': temperature,
        'max_tokens': max_tokens
    }

    url = API_BASE_URL + CHAT_COMPLETIONS_ENDPOINT

    # Debug: Print request details if verbose
    if verbose:
        print(f"Sending POST request to URL: {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")

    while True:
        try:
            response = requests.post(url, headers=headers, json=payload)
            # Debug: Print response details if verbose
            if verbose:
                print(f"Received response with status code: {response.status_code}")
                print(f"Response body: {response.text}")

            if response.status_code == 200:
                response_json = response.json()
                # Extract assistant's reply
                assistant_message = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                if assistant_message:
                    if verbose:
                        # Display full JSON
                        print(json.dumps(response_json, indent=2))
                    else:
                        # Display only assistant's message
                        # Temporarily remove colorization to prevent truncation
                        # If needed, reintroduce colorization carefully
                        print(f"Deepseek AI Response:\n{assistant_message}", flush=True)
                        # Optionally append assistant's reply to history
                        if session_file:
                            history.append({"role": "assistant", "content": assistant_message})
                            # Save updated history
                            save_history(session_file, history)
                else:
                    print("Deepseek AI did not return a message.")
                logging.info(f"Request: {payload}")
                logging.info(f"Response: {response_json}")
                return response_json
            elif response.status_code == 429:
                if handle_rate_limit(response):
                    continue
            elif 400 <= response.status_code < 500:
                print(f"Client Error {response.status_code}: {response.text}")
                logging.error(f"Client Error {response.status_code}: {response.text}")
                return None
            elif 500 <= response.status_code < 600:
                print(f"Server Error {response.status_code}: {response.text}")
                print(f"Retrying in {RATE_LIMIT_RETRY_DELAY} seconds...")
                logging.error(f"Server Error {response.status_code}: {response.text}")
                time.sleep(RATE_LIMIT_RETRY_DELAY)
                continue
            else:
                print(f"Unexpected status code {response.status_code}: {response.text}")
                logging.error(f"Unexpected status code {response.status_code}: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            logging.error(f"Request failed: {e}")
            print(f"Retrying in {RATE_LIMIT_RETRY_DELAY} seconds...")
            time.sleep(RATE_LIMIT_RETRY_DELAY)

def send_message_interactive(api_key, user_input, session_file, model='deepseek-chat', stream=False, temperature=0.2, max_tokens=150, verbose=False, beautify=True):
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
        'model': model,
        'messages': history,
        'stream': stream,
        'temperature': temperature,
        'max_tokens': max_tokens
    }

    url = API_BASE_URL + CHAT_COMPLETIONS_ENDPOINT

    # Debug: Print request details if verbose
    if verbose:
        print(f"Sending POST request to URL: {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")

    while True:
        try:
            response = requests.post(url, headers=headers, json=payload)
            # Debug: Print response details if verbose
            if verbose:
                print(f"Received response with status code: {response.status_code}")
                print(f"Response body: {response.text}")

            if response.status_code == 200:
                response_json = response.json()
                # Extract assistant's reply
                assistant_message = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                if assistant_message:
                    if verbose:
                        # Display full JSON
                        print(json.dumps(response_json, indent=2))
                    else:
                        # Display only assistant's message
                        # Temporarily remove colorization to prevent truncation
                        print(f"Deepseek AI Response:\n{assistant_message}", flush=True)
                        # Append assistant's reply to history
                        history.append({"role": "assistant", "content": assistant_message})
                        # Save updated history
                        save_history(session_file, history)
                else:
                    print("Deepseek AI did not return a message.")
                logging.info(f"Request: {payload}")
                logging.info(f"Response: {response_json}")
                return response_json
            elif response.status_code == 429:
                if handle_rate_limit(response):
                    continue
            elif 400 <= response.status_code < 500:
                print(f"Client Error {response.status_code}: {response.text}")
                logging.error(f"Client Error {response.status_code}: {response.text}")
                return None
            elif 500 <= response.status_code < 600:
                print(f"Server Error {response.status_code}: {response.text}")
                print(f"Retrying in {RATE_LIMIT_RETRY_DELAY} seconds...")
                logging.error(f"Server Error {response.status_code}: {response.text}")
                time.sleep(RATE_LIMIT_RETRY_DELAY)
                continue
            else:
                print(f"Unexpected status code {response.status_code}: {response.text}")
                logging.error(f"Unexpected status code {response.status_code}: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            logging.error(f"Request failed: {e}")
            print(f"Retrying in {RATE_LIMIT_RETRY_DELAY} seconds...")
            time.sleep(RATE_LIMIT_RETRY_DELAY)

def health_check(api_key):
    """Perform a health check on the API."""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    url = API_BASE_URL + "/health"  # Replace with actual health check endpoint if different

    print(f"Sending GET request to URL: {url}")

    try:
        response = requests.get(url, headers=headers)
        print(f"Health Check Status: {response.status_code}")
        print(f"Response: {response.text}")
        logging.info(f"Health Check Response: {response.json() if response.content else response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        logging.error(f"Health check failed: {e}")

def get_usage(api_key):
    """Retrieve API usage information."""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }
    url = API_BASE_URL + USAGE_ENDPOINT

    print(f"Sending GET request to URL: {url}")

    try:
        response = requests.get(url, headers=headers)
        print(f"Usage Status: {response.status_code}")
        if response.status_code == 200:
            usage_info = response.json()
            print("API Usage Information:")
            print(json.dumps(usage_info, indent=2))
            logging.info(f"Usage Info: {usage_info}")
        else:
            print(f"Failed to retrieve usage info. Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            logging.error(f"Failed to retrieve usage info. Status Code: {response.status_code}, Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Usage info request failed: {e}")
        logging.error(f"Usage info request failed: {e}")

def list_models(api_key):
    """List available models."""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }
    url = API_BASE_URL + MODELS_ENDPOINT

    print(f"Sending GET request to URL: {url}")

    try:
        response = requests.get(url, headers=headers)
        print(f"List Models Status: {response.status_code}")
        if response.status_code == 200:
            models_response = response.json()
            # Adjust based on actual API response structure
            models = models_response.get('models') or models_response.get('data') or models_response
            if isinstance(models, list) and models:
                print("Available Models:")
                for model_item in models:
                    if isinstance(model_item, dict):
                        # Adjust the key based on actual model structure
                        model_name = model_item.get('name') or model_item.get('id') or model_item.get('model') or str(model_item)
                        print(f"- {model_name}")
                    else:
                        print(f"- {model_item}")
            else:
                print("No models found.")
            logging.info(f"Models: {models}")
        else:
            print(f"Failed to retrieve models. Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            logging.error(f"Failed to retrieve models. Status Code: {response.status_code}, Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"List models request failed: {e}")
        logging.error(f"List models request failed: {e}")

def interactive_mode(api_key, attach=None, model='deepseek-chat', stream=False, temperature=0.2, max_tokens=150, verbose=False, beautify=True):
    """Start an interactive conversation with Deepseek AI."""
    session_file = None
    session_name = None

    if attach:
        session_file = resolve_session(attach)
        if session_file:
            session_name = session_file.stem
            print(f"Continuing conversation in session '{session_name}'.")
        else:
            print("Starting a new conversation.")
    else:
        print("Starting a new conversation.")

    if not session_file:
        # Create a new session with a timestamp
        session_name = f"session_{int(time.time())}"
        session_file = SESSIONS_DIR / f"{session_name}.json"
        # Initialize with system prompt
        history = [
            {
                "role": "system",
                "content": "You are a highly knowledgeable and accurate assistant. Please provide correct and concise answers to the user's questions."
            }
        ]
        save_history(session_file, history)
        print(f"New session created: '{session_name}'")

    print("Interactive mode. Type 'exit' or 'quit' to end the session.\n")

    # Initialize PromptSession from prompt_toolkit with custom key bindings
    bindings = KeyBindings()

    @bindings.add('c-d')
    def _(event):
        """
        Handle Ctrl+D to send the message instead of exiting.
        """
        buffer = event.app.current_buffer
        buffer.validate_and_handle()

    session = PromptSession(
        history=InMemoryHistory(),
        auto_suggest=AutoSuggestFromHistory(),
        key_bindings=bindings
    )

    while True:
        try:
            user_input = session.prompt(
                "You: ",
                multiline=True,
                bottom_toolbar="Press Ctrl+D to send the message, Ctrl+C to cancel."
            )
            if user_input.strip().lower() in ['exit', 'quit']:
                print("Ending interactive session.")
                break
            if not user_input.strip():
                continue  # Skip empty inputs

            send_message_interactive(
                api_key=api_key,
                user_input=user_input,
                session_file=session_file,
                model=model,
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens,
                verbose=verbose,
                beautify=beautify
            )

        except (KeyboardInterrupt, EOFError):
            print("\nEnding interactive session.")
            break

def main():
    """Main function to handle command-line interactions."""
    initialize_directories()
    setup_logging()
    args = parse_arguments()
    api_key = get_api_key()

    if args.command == 'send':
        session_file = None
        if args.attach:
            session_file = resolve_session(args.attach)
            if session_file:
                print(f"Attaching to session '{session_file.stem}' for context.")
            else:
                print("Proceeding without attaching to a session.")

        response = send_message(
            api_key=api_key,
            message=args.message,
            model=args.model,
            stream=args.stream,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            session_file=session_file,
            verbose=args.verbose,
            beautify=args.beautify
        )
        if not response:
            print("Failed to get a response from Deepseek AI.")

    elif args.command == 'reset':
        reset_history()

    elif args.command == 'history':
        view_history()

    elif args.command == 'list-sessions':
        list_sessions()

    elif args.command == 'health-check':
        health_check(api_key)

    elif args.command == 'usage':
        get_usage(api_key)

    elif args.command == 'list-models':
        list_models(api_key)

    elif args.command == 'interactive':
        interactive_mode(
            api_key=api_key,
            attach=args.attach,
            model=args.model,
            stream=args.stream,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
            beautify=args.beautify
        )

    else:
        print("No command provided. Use -h for help.")

if __name__ == '__main__':
    main()
