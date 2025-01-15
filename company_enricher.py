import pandas as pd
import time
import openai
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def count_tokens(text):
    """Estimate the number of tokens in a text string"""
    # Rough estimation: 1 token ≈ 4 characters for English text
    return len(text) // 4

def truncate_to_token_limit(text, max_tokens=50000):
    """Truncate text to stay within token limit"""
    current_tokens = count_tokens(text)
    if current_tokens <= max_tokens:
        return text
    
    # If we need to truncate, do it by ratio
    ratio = max_tokens / current_tokens
    char_limit = int(len(text) * ratio)
    return text[:char_limit] + "... [truncated]"

def extract_website_data(url):
    """Extract data from company website"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text content (limited to main content areas)
            text_blocks = []
            for tag in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                elements = soup.find_all(tag)
                text_blocks.extend([elem.get_text().strip() for elem in elements if elem.get_text().strip()])
            
            # Combine and truncate text content
            text_content = ' '.join(text_blocks)
            text_content = truncate_to_token_limit(text_content, 40000)  # Leave room for other fields
            
            # Extract meta description
            meta_desc = soup.find('meta', {'name': 'description'})
            description = meta_desc['content'] if meta_desc else ''
            
            # Extract contact information
            emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', response.text)
            
            return {
                'text_content': text_content,
                'meta_description': description,
                'contact_info': emails[:5]  # Limit to first 5 emails
            }
    except Exception as e:
        print(f"Error scraping website: {str(e)}")
    return None

def analyze_website_data(website_data, max_tokens=50000):
    """Analyze website data while respecting token limit"""
    # Combine all relevant data
    combined_text = f"""
    Title: {website_data.get('title', 'N/A')}
    Description: {website_data.get('meta_description', 'N/A')}
    Main Content: {website_data.get('text_content', 'N/A')}
    Contact Info: {', '.join(website_data.get('contact_info', []))}
    """
    
    # Truncate if needed
    truncated_text = truncate_to_token_limit(combined_text, max_tokens)
    
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": "You are an expert at analyzing company information. Provide a detailed analysis of the company based on their website content."
            }, {
                "role": "user",
                "content": f"Analyze this company based on their website content:\n\n{truncated_text}"
            }],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return None

def enrich_company_data(company):
    """Enrich company data with website information"""
    if not company['Website'] or company['Website'] == 'N/A':
        return None
    
    # Extract website data
    website_data = extract_website_data(company['Website'])
    if not website_data:
        return None
    
    # Analyze the data
    analysis = analyze_website_data(website_data)
    
    return {
        'website_description': website_data.get('meta_description', ''),
        'contact_info': website_data.get('contact_info', []),
        'detailed_analysis': analysis if analysis else 'Analysis failed'
    }

def enrich_company_data():
    """Main function to enrich company data"""
    try:
        companies_df = pd.read_csv('createx_companies.csv')
        print("Successfully loaded companies data")
    except Exception as e:
        print(f"Error loading companies data: {str(e)}")
        return
    
    print("Available columns:", companies_df.columns.tolist())
    
    column_mapping = {
        'name': 'Company',
        'website': 'Website'
    }
    companies_df = companies_df.rename(columns=column_mapping)
    
    required_columns = ['Company', 'Website']
    missing_columns = [col for col in required_columns if col not in companies_df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return
    
    output_file = 'enriched_companies.pkl'
    
    while True:  # Keep running until all companies are processed or interrupted
        try:
            if os.path.exists(output_file):
                enriched_df = pd.read_pickle(output_file)
                processed_companies = set(enriched_df['Company'])
                print(f"Found existing data for {len(processed_companies)} companies")
            else:
                enriched_df = pd.DataFrame()
                processed_companies = set()
        except Exception as e:
            print(f"Error loading existing data: {str(e)}")
            enriched_df = pd.DataFrame()
            processed_companies = set()
        
        try:
            # Find unprocessed companies
            unprocessed_companies = companies_df[~companies_df['Company'].isin(processed_companies)]
            total_unprocessed = len(unprocessed_companies)
            
            if total_unprocessed == 0:
                print("All companies have been processed!")
                break
            
            # Select a random unprocessed company
            random_company = unprocessed_companies.sample(n=1).iloc[0]
            print(f"\nRandomly selected company: {random_company['Company']}")
            print(f"Remaining unprocessed companies: {total_unprocessed - 1}")
            
            website_data = None
            if pd.notna(random_company['Website']):
                try:
                    website_data = extract_website_data(random_company['Website'])
                except Exception as e:
                    print(f"Error extracting website data: {str(e)}")
                    website_data = None
            
            analysis = analyze_website_data(website_data) if website_data else "No website data available for analysis."
            
            try:
                new_data = {
                    'Company': random_company['Company'],
                    'Website': random_company.get('Website', ''),
                    'website_title': website_data.get('title', '') if website_data else '',
                    'website_description': website_data.get('meta_description', '') if website_data else '',
                    'website_keywords': website_data.get('meta_keywords', '') if website_data else '',
                    'main_content': website_data.get('main_content', '') if website_data else '',
                    'about_content': website_data.get('about_content', '') if website_data else '',
                    'product_content': website_data.get('product_content', '') if website_data else '',
                    'team_content': website_data.get('team_content', '') if website_data else '',
                    'all_content': website_data.get('all_content', '') if website_data else '',
                    'contact_info': website_data.get('contact_info', []) if website_data else [],
                    'social_links': website_data.get('social_links', []) if website_data else [],
                    'pages_scraped': len(website_data.get('pages_scraped', [])) if website_data else 0,
                    'detailed_analysis': analysis
                }
                
                new_row_df = pd.DataFrame([new_data])
                enriched_df = pd.concat([enriched_df, new_row_df], ignore_index=True)
                
                try:
                    enriched_df.to_pickle(output_file)
                    print(f"Progress saved for {random_company['Company']}")
                    enriched_df.to_csv('enriched_companies.csv', index=False)
                except Exception as e:
                    print(f"Error saving progress: {str(e)}")
                
                print(f"\nCompleted processing company: {random_company['Company']}")
                print(f"Total companies in enriched data: {len(enriched_df)}")
                print(f"Remaining unprocessed companies: {total_unprocessed - 1}")
                
                # Add a delay between companies to avoid overwhelming servers
                time.sleep(3)
                
            except Exception as e:
                print(f"Error processing company data: {str(e)}")
                time.sleep(3)  # Still delay even if there was an error
                continue
        
        except Exception as e:
            print(f"Error in main processing loop: {str(e)}")
            time.sleep(3)  # Delay before retrying
            continue
        
        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
            break

if __name__ == "__main__":
    try:
        enrich_company_data()
    except KeyboardInterrupt:
        print("\nScript terminated by user")
    finally:
        print("Script completed")