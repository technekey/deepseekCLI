**Deepseek CLI** is a powerful command-line interface tool designed to interact seamlessly with the Deepseek AI API. Whether you're looking to send quick queries, engage in interactive conversations, or manage your AI sessions. **This is not at all associated with deepseek project.** This repo provide terminal based client using deepseek API to converse with deepseek API.

Note: You can switch between models and change temperature during the ongoing conversation.



## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Send a Single Message](#send-a-single-message)
  - [Interactive Mode](#interactive-mode)
  - [Attach to an Existing Session](#attach-to-an-existing-session)
  - [List Available Models](#list-available-models)
  - [Check API Usage](#check-api-usage)
  - [Reset Conversation Histories](#reset-conversation-histories)
  - [View Conversation Histories](#view-conversation-histories)
- [Examples](#examples)
- [Logging](#logging)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Send Messages:** Quickly send messages to Deepseek AI and receive responses.
- **Interactive Conversations:** Engage in ongoing conversations with context preservation.
- **Session Management:** Create, attach, and manage multiple conversation sessions.
- **Model Listing:** View all available AI models supported by Deepseek.
- **API Usage Monitoring:** Track your API usage and monitor your account balance.
- **Conversation History:** Save and view past conversations for reference.
- **Health Checks:** Verify the connectivity and status of the Deepseek API.
- **Logging:** Comprehensive logging of all requests and responses for debugging and auditing purposes.

## Prerequisites

- **Python 3.7 or higher**
- **pip** (Python package manager)
- **Deepseek AI API Key**

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/technekey/deepseekCLI.git
   cd deepseekCLI
    ```

2. **Setup virtual env**(OPTIONAL)
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
5. **Make the Script Executable**

   ```
   chmod +x deepseek.py
   ```

6. Setup the API KEY as Environment variable
   ```bash
   export DEEPSEEK_API='your-deepseek-api-key'
   ```
   

Done, you are ready to use this.

# Usage
Deepseek CLI offers various commands to interact with the Deepseek AI API. Below are the primary comm## Send a Single Message
Send a standalone message to Deepseek AI without attaching to any session.
```bash
python3 deepseek.py send "Hello, how are you?"
```
**Options:**
- `--model`: Specify the AI model to use. Defaults to `deepseek-chat`.
- `--stream`: Enable streaming responses from Deepseek AI.
- `--temperature`: Set the sampling temperature for response generation. Higher values make output more optimized.
- `--max_tokens`: Define the maximum number of tokens in the response. Defaults to `8192`.
- `--attach`: Attach to an existing session by name or number for context.
- `--verbose`: Display the full JSON response from Deepseek AI.
- `--beautify`: Enable or disable colorized output. Defaults to `true`. Set to `false` to disable.
- `--config <path>`: Specify a custom configuration file

**Example:**
```bash
python3 deepseek.py send "Explain the concept of recursion in Python." --max_tokens 500
```
## Interactive Mode...similar to web UI expereince
Start an interactive chat session with Deepseek AI, allowing for ongoing conversations with context preserv
```bash
python3 deepseek.py interactive
```
**Options:**
- `--attach`: Attach to an existing session by name or number. If not provided, a new session is initiated.
- `--model`: Specify the AI model to use. Defaults to `deepseek-chat`.
- `--stream`: Enable streaming responses from Deepseek AI.
- `--temperature`: Set the sampling temperature for response generation. Higher values make output more - `--max_tokens`: Define the maximum number of tokens in the response. Defaults to `8192`.
- `--verbose`: Display the full JSON response from Deepseek AI.
- `--beautify`: Enable or disable colorized output. Defaults to `true`. Set to `false` to disable.
**Inside Interactive Mode:**
- **Send Messages:** Type your message and press `Ctrl+D` to send.
- **Exit Session:** Type `exit` or `quit` and press `Enter` to end the session.
**Example:**
```
Interactive mode. Type 'exit' or 'quit' to end the session.
You: Can you help me optimize my Python code?
Deepseek AI Response:
[Assistant's detailed response]
```
## Attach to an Existing Session
Continue a previously saved conversation session to maintain context.

```
python3 deepseek.py interactive --attach 1
```
*Assuming `1` is the session number or name.*

## List Available Models
View all AI models supported by Deepseek AI.

```bash
python3 deepseek.py list-models
```
## Check API Usage
Monitor your API usage and account balance.
```bash
python3 deepseek.py usage
```
## Reset Conversation Histories
Delete all saved conversation histories.
```bash
python3 deepseek.py reset
```
## View Conversation Histories
List and view saved conversation histories.
```bash
python3 deepseek.py history
```
---
# Examples
## Sending a Message with Increased Token Limit
Ensure you receive detailed responses by increasing the `max_tokens` parameter.
```bash
python3 deepseek.py send "Explain the concept of polymorphism in object-oriented programming." max_tokens 150
```
## Starting a New Interactive Session
Begin a fresh conversation with Deepseek AI.
```bash
python3 deepseek.py interactive
```
**Inside Interactive Mode:**
```
Interactive mode. Type 'exit' or 'quit' to end the session.
You: What are the benefits of using design patterns in software development?
Deepseek AI Response:
[Assistant's detailed response]
```

## Attaching to an Existing Session Named 'project_discussion'
Continue an ongoing conversation stored under the session name `project_discussion`.
```bash
python3 deepseek.py interactive --attach project_discussion
```
## Listing All Available AI Models
Retrieve and view all AI models available for use.
```bash
python3 deepseek.py list-models
```
**Sample Output:**
```
Available Models:
- deepseek-chat
- deepseek-reasoner
```
## Checking Current API Usage
Monitor how much of your API quota you've utilized.
```bash
python3 deepseek.py usage
```
**Sample Output:**
```
API Usage Information:
{
 "prompt_tokens": 1500,
 "completion_tokens": 3000,
 "total_tokens": 4500,
 "balance": 50000
}
```
## Resetting All Conversation Histories
Remove all saved conversations to start fresh.
```bash
python3 deepseek.py reset
```
**Sample Output:**
```
All conversation histories have been reset.
```
## Viewing a Specific Conversation History
List all sessions and view the history of a selected session.
```bash
python3 deepseek.py history
```
**Sample Interaction:**
```
Available Sessions:
1. session_1618033988
2. project_discussion
3. brainstorming_session
Enter the session number to view its history (or press Enter to return): 2
Conversation History for Session 'project_discussion':
System: You are a highly knowledgeable and accurate assistant. Please provide correct and concise answUser: How can we improve our project's scalability?
Assistant: [Assistant's detailed response]
```
## Hotkeys

```
CTRL + T : To change the temperature
CTRL + X : To change the model. 
CTRL + D : To send the message.
CTRL + C : To exit. 
```

## Temperature 

```
USECASE	TEMPERATURE
Coding / Math   	0.0
Data Cleaning / Data Analysis	1.0
General Conversation	1.3
Translation	1.3
Creative Writing / Poetry	1.5
```
