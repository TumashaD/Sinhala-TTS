
## Setup Instructions

1. **Ensure Python 3.10 is installed:**
    - On macOS/Linux:
      ```bash
      python3 --version
      ```
      If Python 3.10 is not installed, use a package manager to install it:
      - macOS (Homebrew):
        ```bash
        brew install python@3.10
        ```
      - Ubuntu/Debian:
        ```bash
        sudo apt update
        sudo apt install python3.10 python3.10-venv
        ```
    - On Windows:
      - Download and install Python 3.10 from [python.org](https://www.python.org/downloads/release/python-3100/).

2. **Create a virtual environment using Python 3.10:**
    ```bash
    python3.10 -m venv venv
    ```

3. **Activate the virtual environment:**
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```
    - On Windows:
      ```bash
      venv\Scripts\activate
      ```

4. **Install TTS:**
    ```bash
    pip install TTS
    ```
