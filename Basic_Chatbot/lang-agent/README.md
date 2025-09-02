# Two-Phased Thinking Raging Bot

This project implements a sophisticated chatbot that utilizes a "two-phased thinking" process to provide accurate and context-aware responses. The bot is specialized in physics-related queries and leverages a persistent memory to maintain conversation history.

## How it Works

The core of this project is the "Two-Phased Thinking" agent. This approach separates the agent's process into two distinct phases:

1.  **Reasoning Phase:** When a user asks a question, the agent first enters a reasoning phase. In this phase, it analyzes the query, breaks it down, and decides if it needs to use any tools (like retrieving information from its knowledge base). It formulates a plan to answer the question.
2.  **Response Generation Phase:** Once the agent has a plan and has gathered the necessary information, it generates a coherent and human-like response.

This separation allows the agent to "think" before it "speaks," leading to more thoughtful and accurate answers.

The project is divided into a backend server that powers the agent and a frontend web application for user interaction.

## Project Structure

-   `/src`: Contains the Python-based backend built with FastAPI.
    -   `FastApi.py`: The main entry point for the backend server.
    -   `agent.py`: Defines the core logic of the Two-Phased Thinking agent.
    -   `tools.py`: Contains tools that the agent can use (e.g., for information retrieval).
    -   `memory_manager.py`: Manages the conversation history and memory.
-   `requirements.txt`: Lists all the Python dependencies for the backend.
-   `/frontend/hello/lang-bot`: Contains the Next.js frontend application.
    -   `app/`: The main application pages.
    -   `package.json`: Lists the Node.js dependencies and scripts.

## Getting Started

### Backend Setup

1.  **Install the required Python packages from the `lang-agent` directory:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Navigate to the source directory:**
    ```bash
    cd src
    ```

3.  **Run the FastAPI server:**
    ```bash
    uvicorn FastApi:app --reload
    ```
    The backend server will be running at `http://127.0.0.1:8000`.

### Frontend Setup

1.  **Navigate to the frontend directory from the `lang-agent` root:**
    ```bash
    cd frontend/hello/lang-bot
    ```

2.  **Install the required Node.js packages:**
    ```bash
    npm install
    ```

3.  **Run the Next.js development server:**
    ```bash
    npm run dev
    ```
    The frontend application will be accessible at `http://localhost:3000`. You can now open this URL in your browser to interact with the bot.
