# Expert Knowledge Worker

This project is an Expert Knowledge Worker that uses LangChain, Chroma, and OpenAI's GPT-4o-mini model to load documents, create a vector store, visualize the vector store, and provide a conversational interface.
This project used RAG (Retrieval Augmented Generation) to ensure our question/answering assistant has high accuracy based on local knowledge base.

## Project Preview
[screen-capture (3).webm](https://github.com/user-attachments/assets/c60a98ba-7c74-4790-bb97-1373070ac5f9)

## Project Structure

- `load_documents.py`: Handles loading documents from the knowledge base.
- `vector_store.py`: Handles creating and managing the vector store.
- `visualize_vector_store.py`: Handles visualizing the vector store.
- `chat.py`: Handles the chat functionality.
- `main.py`: The main entry point that uses the other modules.

## Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/your-username/expert-assistant.git
   cd expert-assistant
   ```

2. Create a virtual environment:

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```sh
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add your OpenAI API key:

   ```env
   OPENAI_API_KEY=your-openai-api-key
   ```

### Running the Project

1. Run the main script:

   ```sh
   python main.py
   ```

2. The Gradio interface will launch in your browser.

## Usage

- Load documents from the `knowledge-base` directory.
- Create a vector store from the document chunks.
- Visualize the vector store in 2D and 3D.
- Interact with the expert assistant using the Gradio chat interface.

## License

This project is licensed under the MIT License.
