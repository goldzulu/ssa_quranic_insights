# Quranic Insights: Islamic AI Assistant

A Streamlit app that helps users search and understand the Quran through advanced AI tools.

## Features

- **Surah Lookup**: Retrieve specific surahs by name or number
- **Verse Lookup**: Find specific verses by surah and verse number
- **Juz Lookup**: Search by juz or juz ranges
- **Semantic Search**: Find verses related to specific topics
- **Conversational Interface**: Interact with the app in natural language

## Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Quranic_Insights.git
   cd Quranic_Insights
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Deploying to Streamlit Cloud

1. Push your code to GitHub (make sure the `.env` file is in `.gitignore`)

2. Create a Streamlit account and connect to your GitHub repository

3. Set up the secret in the Streamlit Cloud dashboard:
   - Go to your app settings in Streamlit Cloud
   - Navigate to "Secrets" section
   - Add the following to your secrets:
     ```toml
     OPENAI_API_KEY = "your_openai_api_key_here"
     ```

4. Deploy your app through the Streamlit Cloud interface

## Security Notes

- Never commit your API keys to the repository
- Always use Streamlit secrets or environment variables for sensitive information
- The app does not store user queries or conversation history beyond the current session

## Data Sources

The Quran dataset is automatically downloaded from a GitHub repository if not found locally. The dataset includes:
- Surah names in English and Arabic
- Verse texts in Arabic and English
- Additional metadata like juz numbers, revelation place, etc.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```

### Explanation:
- **Introduction and Features**: An overview of what the app does.
- **Local Development**: Provides instructions for setting up the app locally.
- **Deploying to Streamlit Cloud**: Explains how to deploy the app to Streamlit Cloud.
- **Security Notes**: Provides security considerations for handling sensitive information.
- **Data Sources**: Describes the data sources used by the app.
- **License**: Credits and licensing information.

This `README.md` file should give users a clear understanding of how to set up and use your app.
