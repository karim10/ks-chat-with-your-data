# Tech Talk: Chat with Your Data

**Date:** 4th December 2024

## Resources

- **Slides:** [Google Slides](https://docs.google.com/presentation/d/1YIrwaAGIwAwqec_A7YrBVtM882Xlv-iU3qjOBskttPs)
- **Recording:** [Google Drive](https://drive.google.com/file/d/1CzJsMVJpo6P6YaF8EPn5IGeqIrya-jvc/view?usp=sharing)

## Prerequisites

- **Python:** Version 3.10+
- **Ollama**

## Setup Instructions

### Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the App

- Add your teams members profile pdfs to the `profiles` folder.
- They should be named like `Karim_BenYezza.pdf`.

```bash
python team_chat_bot.py
```
