# StructHunt

## Overview

StructHunt is a program designed to scrape scientific articles from BioRXiv, parse them, convert them into embeddings, and perform analysis on whether they employ certain methodologies. The resulting information is then organized and stored in a CSV file. The program consists of several components that work together seamlessly to achieve this functionality.

## Components

### 1. `scraper.py`

`scraper.py` is responsible for scraping BioRXiv to obtain scientific articles in PDF format. It utilizes external libraries and APIs to download these articles and then applies the necessary parsing logic to extract relevant information.

### 2. `VectorDatabase.py`

`VectorDatabase.py` contains the `Lantern` class, which is used to interact with a PostgreSQL database. The embeddings generated from the articles are input into the database, associating them with the corresponding publications.

### 3. `runner.py`

`runner.py` is the script responsible for managing the overall flow of the program. It identifies publications that haven't been processed, retrieves their IDs, and triggers subsequent processing steps.

### 4. `chatgpt`

The `chatgpt` component involves interacting with OpenAI's GPT-based language model. This is done using prompts generated from the `updated_prompt.py` script along with the embeddings retrieved from the previous step. The goal is to analyze whether the publications implement certain methodologies.

### 5. `updated_prompt.py`

`updated_prompt.py` generates prompts that are used to query the GPT model. These prompts are crafted based on the specific characteristics of the publications being analyzed.

### 6. `CSV Output`

The program populates a CSV file with the analysis results. This file contains information on whether the publications employ certain methodologies, providing a structured output for easy interpretation and further analysis.

## Getting Started

1. **Environment Setup:**
    - Ensure that you have Python installed.
    - Install the required Postgres Database and Python packages using `initialize_database.sh`.

    ```bash
    sudo ./initialize_database.sh
    ```

2. **Run the Program:**
    - Execute `runner.py` to initiate the structured hunting process.

```bash
python runner.py
```

## Contributing

Feel free to contribute to the development of StructHunt by submitting issues, feature requests, or pull requests. Your feedback and contributions are highly appreciated.

## License

This project is licensed under the [MIT License](LICENSE).