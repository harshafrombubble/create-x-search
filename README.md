# Create-X Company and Founder Search

An interactive search platform for Create-X companies and founders, powered by AI. This application provides semantic search capabilities, deep dive analysis of companies, and founder information tracking.

## Features

- 🔍 **Semantic Search**: Search companies and founders using natural language
- 🏢 **Company Deep Dives**: Get detailed analysis of companies including website content
- 👥 **Founder Insights**: Access founder backgrounds and LinkedIn profiles
- 📊 **Live Statistics**: Track visitor engagement and query patterns
- 💬 **Chat Interface**: User-friendly chat-based interaction
- 🔄 **Real-time Updates**: Live data processing and analysis

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up secrets:
   - Create a `.streamlit` directory in your project root
   - Create `secrets.toml` inside `.streamlit` directory with:
```toml
# OpenAI API Configuration
openai_api_key = "your-api-key-here"

# LinkedIn Credentials (if needed)
linkedin_username = "your-linkedin-email"
linkedin_password = "your-linkedin-password"
```

5. Run the application:
```bash
streamlit run query_interface.py
```

## Usage

### Basic Search
- Type natural language queries about companies or founders
- Use industry terms, technologies, or other relevant keywords
- View matched companies and founders with detailed information

### Deep Dive Analysis
- Request detailed analysis of specific companies
- Format: "deep dive [Company Name]"
- Get comprehensive insights including website analysis

### Example Queries
- "Show me companies in the healthcare industry"
- "Find founders with experience in artificial intelligence"
- "Deep dive analysis of [Company Name]"
- "What companies are focused on B2B solutions?"
- "Find founders who studied at Georgia Tech"

## Data Structure

The application uses several data files:
- `enriched_companies.pkl`: Processed company data
- `enriched_founders.pkl`: Processed founder data
- `createx_search.db`: SQLite database for visitor tracking
- `visitor_stats.csv`: Visitor statistics
- `stored_queries.csv`: Query history

## Dependencies

- Streamlit
- OpenAI GPT-4
- Pandas
- BeautifulSoup4
- SQLite
- Scikit-learn
- Selenium (for LinkedIn data)

## Notes

- The application requires an active internet connection
- Some features use web scraping, which may be affected by website changes
- API rate limits may apply for OpenAI services
- Never commit your `secrets.toml` file to version control

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 