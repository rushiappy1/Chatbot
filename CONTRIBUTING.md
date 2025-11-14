# Contributing

Thanks for your interest in improving this RAG chatbot project!

## Ways to contribute

- Report bugs or issues you encounter.
- Propose and implement new features (incremental indexing, auth, better UI, etc.).
- Improve documentation, examples, and diagrams.
- Add or improve tests.

## Development setup

1. Fork the repository and clone your fork.
2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\\Scripts\\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Run the app locally:

   ```bash
   streamlit run app.py --server.port 7860
   ```

4. Run tests (once they are added):

   ```bash
   pytest
   ```

## Coding style

- Prefer clear, readable Python.
- Keep functions small and focused.
- Add docstrings for public functions.
- Avoid breaking changes to the public behavior of the chatbot without documenting them.

## Pull requests

- Create a feature branch for your change.
- Make sure the app runs and tests pass.
- Describe **what** you changed and **why** in the PR description.
- If your change affects behavior or setup, update the README as well.

By contributing, you agree that your contributions will be licensed under the same license as this project (MIT).
