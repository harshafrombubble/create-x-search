import streamlit as st
import pandas as pd
import openai
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import os
from founder_enricher import extract_linkedin_data, analyze_founder_background, setup_driver, linkedin_login
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
import hashlib
import logging
import traceback
import sqlite3
from contextlib import contextmanager
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse

# Set up logging first
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('search_errors.log'),
        logging.StreamHandler()
    ]
)

def init_db():
    """Initialize SQLite database with necessary tables"""
    with sqlite3.connect('createx_search.db') as conn:
        cursor = conn.cursor()
        
        # Create visitors table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS visitors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            visitor_id TEXT NOT NULL,
            date DATE NOT NULL,
            visit_count INTEGER DEFAULT 1,
            queries INTEGER DEFAULT 0
        )
        ''')
        
        # Create queries table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            query TEXT NOT NULL,
            response TEXT,
            num_companies_returned INTEGER,
            num_founders_returned INTEGER
        )
        ''')
        
        conn.commit()

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect('createx_search.db')
    try:
        yield conn
    finally:
        conn.close()

def track_visitor():
    """Track unique visitors using session cookies and store in SQLite"""
    if 'visitor_id' not in st.session_state:
        unique_id = f"{datetime.now().timestamp()}-{np.random.randint(1000000)}"
        st.session_state.visitor_id = hashlib.md5(unique_id.encode()).hexdigest()
    
    today = datetime.now().date()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Check if visitor exists for today
        cursor.execute('''
            SELECT id, visit_count FROM visitors 
            WHERE visitor_id = ? AND date = ?
        ''', (st.session_state.visitor_id, today))
        
        result = cursor.fetchone()
        
        if result:
            # Update existing visit
            cursor.execute('''
                UPDATE visitors 
                SET visit_count = visit_count + 1 
                WHERE visitor_id = ? AND date = ?
            ''', (st.session_state.visitor_id, today))
        else:
            # New visit
            cursor.execute('''
                INSERT INTO visitors (visitor_id, date, visit_count, queries)
                VALUES (?, ?, 1, 0)
            ''', (st.session_state.visitor_id, today))
        
        conn.commit()

def update_query_count():
    """Update query count for current visitor"""
    today = datetime.now().date()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE visitors 
            SET queries = queries + 1 
            WHERE visitor_id = ? AND date = ?
        ''', (st.session_state.visitor_id, today))
        conn.commit()

def get_visitor_stats():
    """Get summary statistics for visitors from SQLite"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get total unique visitors
        cursor.execute('SELECT COUNT(DISTINCT visitor_id) FROM visitors')
        total_visitors = cursor.fetchone()[0]
        
        # Get total visits
        cursor.execute('SELECT SUM(visit_count) FROM visitors')
        total_visits = cursor.fetchone()[0] or 0
        
        # Get total queries
        cursor.execute('SELECT SUM(queries) FROM visitors')
        total_queries = cursor.fetchone()[0] or 0
        
        # Get today's visitors
        today = datetime.now().date()
        cursor.execute('''
            SELECT COUNT(DISTINCT visitor_id) 
            FROM visitors 
            WHERE date = ?
        ''', (today,))
        today_visitors = cursor.fetchone()[0]
        
        return {
            'total_visitors': total_visitors,
            'total_visits': total_visits,
            'total_queries': total_queries,
            'today_visitors': today_visitors
        }

def store_query(query, response, num_companies=0, num_founders=0):
    """Store query and response in SQLite database"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO queries 
            (timestamp, query, response, num_companies_returned, num_founders_returned)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now(), query, response, num_companies, num_founders))
        conn.commit()

def load_data():
    """Load both enriched datasets"""
    try:
        companies_df = pd.read_pickle('enriched_companies.pkl')
        founders_df = pd.read_pickle('enriched_founders.pkl')
        return companies_df, founders_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def clean_text_field(text):
    """Clean text fields to prevent corruption"""
    if pd.isna(text):
        return ""
    text = str(text)
    # More aggressive cleaning for names to prevent corruption
    if '.' * 5 in text:  # If text contains 5 or more consecutive periods
        # Take only the first part before repeated periods
        text = text.split('.')[0]
    # Remove repeated patterns that might indicate corruption
    if len(text) > 1000 or text.count('.') > 20:  # Likely corrupted if too long or too many periods
        return text[:100]  # Return truncated version
    return text.strip()

def search_relevant_items(query, companies_data, founders_data, top_k=5):
    """Find relevant companies and founders using semantic search"""
    try:
        relevant_companies = pd.DataFrame()
        relevant_founders = pd.DataFrame()
        
        if companies_data is not None and len(companies_data) > 0:
            # Clean company text fields
            company_texts = companies_data.apply(
                lambda x: ' '.join(filter(None, [
                    clean_text_field(x['Company']),
                    clean_text_field(x.get('Description', '')),
                    clean_text_field(x.get('Industry', '')),
                    clean_text_field(x.get('Market_Segment', '')),
                    clean_text_field(x.get('detailed_analysis', ''))
                ])), 
                axis=1
            )
            
            # Create TF-IDF vectors with bigrams and trigrams to capture phrases
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 3),
                max_features=5000
            )
            
            try:
                company_vectors = vectorizer.fit_transform(company_texts)
                query_vector = vectorizer.transform([query])
                
                # Calculate similarities
                similarities = cosine_similarity(query_vector, company_vectors)[0]
                
                # Get indices of companies with similarity > 0.1
                relevant_indices = np.where(similarities > 0.1)[0]
                
                if len(relevant_indices) > 0:
                    # Sort by similarity score
                    sorted_indices = relevant_indices[np.argsort(similarities[relevant_indices])[::-1]]
                    
                    # Take top-k results
                    top_indices = sorted_indices[:top_k]
                    
                    # Create DataFrame with relevant companies and their scores
                    relevant_companies = companies_data.iloc[top_indices].copy()
                    relevant_companies['similarity_score'] = similarities[top_indices]
                    
            except Exception as e:
                st.error(f"Error in company search: {str(e)}")
                relevant_companies = pd.DataFrame()
        
        if founders_data is not None and len(founders_data) > 0:
            # Clean founder text fields
            founder_texts = founders_data.apply(
                lambda x: ' '.join(filter(None, [
                    clean_text_field(x['Name']),
                    clean_text_field(x.get('Company', '')),
                    clean_text_field(x.get('linkedin_headline', '')),
                    clean_text_field(x.get('detailed_analysis', ''))
                ])), 
                axis=1
            )
            
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 3),
                max_features=5000
            )
            
            try:
                founder_vectors = vectorizer.fit_transform(founder_texts)
                query_vector = vectorizer.transform([query])
                
                # Calculate similarities
                similarities = cosine_similarity(query_vector, founder_vectors)[0]
                
                # Get indices of founders with similarity > 0.1
                relevant_indices = np.where(similarities > 0.1)[0]
                
                if len(relevant_indices) > 0:
                    # Sort by similarity score
                    sorted_indices = relevant_indices[np.argsort(similarities[relevant_indices])[::-1]]
                    
                    # Take top-k results
                    top_indices = sorted_indices[:top_k]
                    
                    # Create DataFrame with relevant founders and their scores
                    relevant_founders = founders_data.iloc[top_indices].copy()
                    relevant_founders['similarity_score'] = similarities[top_indices]
                    
            except Exception as e:
                st.error(f"Error in founder search: {str(e)}")
                relevant_founders = pd.DataFrame()
        
    except Exception as e:
        logging.error(f"Error in search_relevant_items: {str(e)}\n{traceback.format_exc()}")
        return pd.DataFrame(), pd.DataFrame()
    
    return relevant_companies, relevant_founders

def perform_deep_dive(linkedin_url, is_founder=True):
    """Perform a deep dive analysis using LinkedIn data"""
    try:
        driver = setup_driver()
        if not linkedin_login(driver):
            st.error("Failed to login to LinkedIn")
            driver.quit()
            return None
        
        profile_data = extract_linkedin_data(driver, linkedin_url)
        driver.quit()
        
        if profile_data:
            if is_founder:
                analysis = analyze_founder_background(profile_data)
            else:
                # For companies, we might want to modify the analysis prompt
                analysis = analyze_founder_background(profile_data)  # Using same function for now
            return analysis
        else:
            return "Failed to extract LinkedIn data"
    except Exception as e:
        st.error(f"Error in deep dive analysis: {str(e)}")
        return None
    finally:
        try:
            driver.quit()
        except:
            pass

def scrape_website(url):
    """Scrape website content using BeautifulSoup"""
    try:
        # Add headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Get the main page
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text(separator='\n')
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Get all internal links
        internal_links = []
        domain = urlparse(url).netloc
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(url, href)
            if urlparse(full_url).netloc == domain:
                internal_links.append(full_url)
        
        # Sample a few internal pages (up to 3)
        additional_text = ""
        visited_links = set()
        for link in internal_links[:3]:
            if link not in visited_links:
                try:
                    sub_response = requests.get(link, headers=headers, timeout=5)
                    sub_soup = BeautifulSoup(sub_response.text, 'html.parser')
                    for script in sub_soup(["script", "style"]):
                        script.decompose()
                    sub_text = sub_soup.get_text(separator='\n')
                    additional_text += f"\n\nContent from {link}:\n" + sub_text
                    visited_links.add(link)
                except Exception as e:
                    logging.error(f"Error scraping internal link {link}: {str(e)}")
        
        return text + additional_text
    except Exception as e:
        logging.error(f"Error scraping website {url}: {str(e)}")
        return None

def analyze_website_content(url, query):
    """Analyze website content using GPT-4"""
    try:
        content = scrape_website(url)
        if not content:
            return "Failed to scrape website content."
        
        # Truncate content if too long (GPT-4 context limit)
        if len(content) > 12000:
            content = content[:12000] + "..."
        
        prompt = f"""Analyze the following website content in relation to this query: {query}

Website Content:
{content}

Instructions:
1. Focus on information relevant to the query
2. Extract key details about products, services, or technology
3. Identify any unique selling points or innovations
4. Note any relevant partnerships or achievements
5. Highlight information about the team or company culture
6. Summarize the company's market focus and target audience

Please provide a detailed but concise analysis."""

        # Initialize OpenAI client
        client = openai.OpenAI(api_key=st.secrets["openai_api_key"])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error analyzing website: {str(e)}")
        return f"Error analyzing website: {str(e)}"

def extract_company_name(query, companies_data):
    """Use GPT to extract company name from query and match it to database"""
    try:
        # First try direct matching with company names
        companies_list = companies_data['Company'].dropna().tolist()
        
        # Create prompt for GPT
        companies_context = "\n".join(companies_list)
        prompt = f"""Given this query: "{query}"
And this list of companies:
{companies_context}

Task: If this query is asking for a deep dive or detailed analysis of a company, identify which company from the list is being referred to.
If multiple companies could match, list them all.
If no company matches or the query isn't asking for a company analysis, return "None".
Return ONLY the exact company name(s) from the list, or "None". Do not add any other text."""

        # Initialize OpenAI client
        client = openai.OpenAI(api_key=st.secrets["openai_api_key"])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        
        extracted_companies = response.choices[0].message.content.strip()
        if extracted_companies.lower() == "none":
            return None
            
        # Split in case multiple companies were returned
        possible_companies = [company.strip() for company in extracted_companies.split('\n')]
        
        # Verify each company exists in our database
        verified_companies = []
        for company in possible_companies:
            if not companies_data[companies_data['Company'].str.lower() == company.lower()].empty:
                verified_companies.append(company)
        
        if len(verified_companies) == 1:
            return verified_companies[0]
        elif len(verified_companies) > 1:
            # If multiple matches, return them all for user selection
            return verified_companies
        return None
        
    except Exception as e:
        logging.error(f"Error in company name extraction: {str(e)}")
        return None

def query_data(query, companies_data, founders_data, conversation_history):
    """Query the enriched data using GPT"""
    try:
        if companies_data is None or founders_data is None:
            return "No data available to search."
        
        # Check if this might be a deep dive request
        if 'deep dive' in query.lower() or 'analyze' in query.lower() or 'analysis' in query.lower():
            extracted_result = extract_company_name(query, companies_data)
            
            if extracted_result is None:
                # No company found, proceed with normal search
                pass
            elif isinstance(extracted_result, list):
                # Multiple companies found
                return f"I found multiple possible companies for your deep dive request. Please specify which one you'd like to analyze:\n" + "\n".join([f"- {company}" for company in extracted_result])
            else:
                # Single company found, proceed with deep dive
                company_name = extracted_result
                company_match = companies_data[companies_data['Company'].str.lower() == company_name.lower()]
                
                if company_match.empty:
                    return f"Company '{company_name}' not found in our database. Please check the company name and try again."
                
                company = company_match.iloc[0]
                website = company.get('Website') if 'Website' in company.index and isinstance(company['Website'], str) else None
                
                if not website:
                    return f"No website information available for {company_name}."
                
                # Clean up URL
                if not website.startswith(('http://', 'https://')):
                    website = 'https://' + website
                
                # Get website analysis
                website_analysis = analyze_website_content(website, query)
                
                # Combine with company information from database
                company_info = []
                company_info.append(f"Company: {company.get('Company', 'N/A')}")
                
                if 'website_description' in company.index and isinstance(company['website_description'], str):
                    company_info.append(f"\nDatabase Description: {clean_text_field(company['website_description'])}")
                
                if 'detailed_analysis' in company.index and isinstance(company['detailed_analysis'], str):
                    company_info.append(f"\nPrevious Analysis: {clean_text_field(company['detailed_analysis'])}")
                
                # Find founders
                company_founders = founders_data[
                    founders_data['Company'].str.lower() == company_name.lower()
                ]
                
                if not company_founders.empty:
                    founder_list = []
                    for _, founder in company_founders.iterrows():
                        founder_info = []
                        name = founder.get('Name')
                        if isinstance(name, str):
                            founder_info.append(clean_text_field(name))
                        if 'linkedin_url' in founder.index and isinstance(founder['linkedin_url'], str):
                            founder_info.append(f"[LinkedIn]({founder['linkedin_url']})")
                        if 'linkedin_headline' in founder.index and isinstance(founder['linkedin_headline'], str):
                            founder_info.append(f"- {clean_text_field(founder['linkedin_headline'])}")
                        founder_list.append(" ".join(founder_info))
                    company_info.append("\nFounders:\n- " + "\n- ".join(founder_list))
                
                # Combine all information
                full_analysis = f"""Deep Dive Analysis for {company_name}:

Company Information:
{chr(10).join(company_info)}

Website Analysis:
{website_analysis}

Note: You can also request a deep dive analysis on any of the founders mentioned above."""

                return full_analysis
        
        # Find relevant items using semantic search
        relevant_companies, relevant_founders = search_relevant_items(query, companies_data, founders_data)
        
        # Create context based on available data
        company_context = ""
        if not relevant_companies.empty:
            company_info = []
            for idx, company in relevant_companies.iterrows():
                info = [f"Company: {company.get('Company', 'N/A') if isinstance(company.get('Company'), str) else 'N/A'}"]
                
                # Add website and contact information more prominently
                contact_info = []
                
                # Safely check for Website
                if 'Website' in company.index and isinstance(company['Website'], str):
                    contact_info.append(f"Website: {company['Website']}")
                
                # Safely check for contact_info
                if 'contact_info' in company.index and isinstance(company['contact_info'], str):
                    contact_info.append(f"Contact Information: {clean_text_field(company['contact_info'])}")
                
                # Safely check for Email
                if 'Email' in company.index and isinstance(company['Email'], str):
                    contact_info.append(f"Email: {company['Email']}")
                
                if contact_info:
                    info.append("Contact Details:\n- " + "\n- ".join(contact_info))
                
                # Find and list ALL founders for this company - using case-insensitive comparison
                company_name = company.get('Company')
                if isinstance(company_name, str):
                    company_founders = founders_data[
                        founders_data['Company'].str.lower() == company_name.lower()
                    ]
                else:
                    company_founders = pd.DataFrame()
                
                if not company_founders.empty:
                    founder_list = []
                    for _, founder in company_founders.iterrows():
                        founder_info = []
                        
                        # Safely get founder name
                        name = founder.get('Name')
                        if isinstance(name, str):
                            founder_info.append(clean_text_field(name))
                        else:
                            founder_info.append('N/A')
                        
                        # Safely get LinkedIn URL
                        if 'linkedin_url' in founder.index and isinstance(founder['linkedin_url'], str):
                            founder_info.append(f"[LinkedIn]({founder['linkedin_url']})")
                        
                        # Safely get LinkedIn headline
                        if 'linkedin_headline' in founder.index and isinstance(founder['linkedin_headline'], str):
                            founder_info.append(f"- {clean_text_field(founder['linkedin_headline'])}")
                        
                        founder_list.append(" ".join(founder_info))
                    info.append("Founders:\n- " + "\n- ".join(founder_list))
                else:
                    info.append("Founders: No founder information available")
                
                # Add company description and other details
                if 'website_description' in company.index and isinstance(company['website_description'], str):
                    info.append(f"Description: {clean_text_field(company['website_description'])}")
                if 'detailed_analysis' in company.index and isinstance(company['detailed_analysis'], str):
                    info.append(f"Analysis: {clean_text_field(company['detailed_analysis'])}")
                
                company_info.append("\n".join(info))
            
            company_context = "\nRelevant Companies:\n" + "\n\n".join(company_info)

        founder_context = ""
        if not relevant_founders.empty:
            founder_info = []
            for idx, founder in relevant_founders.iterrows():
                info = []
                
                # Safely get founder name
                name = founder.get('Name')
                if isinstance(name, str):
                    info.append(f"Name: {name}")
                else:
                    info.append("Name: N/A")
                
                # Safely get company
                if 'Company' in founder.index and isinstance(founder['Company'], str):
                    info.append(f"Company: {founder['Company']}")
                
                # Safely get LinkedIn URL
                if 'linkedin_url' in founder.index and isinstance(founder['linkedin_url'], str):
                    info.append(f"LinkedIn: {founder['linkedin_url']}")
                
                # Safely get LinkedIn headline
                if 'linkedin_headline' in founder.index and isinstance(founder['linkedin_headline'], str):
                    info.append(f"Headline: {clean_text_field(founder['linkedin_headline'])}")
                
                # Safely get LinkedIn summary
                if 'linkedin_summary' in founder.index and isinstance(founder['linkedin_summary'], str):
                    info.append(f"Summary: {clean_text_field(founder['linkedin_summary'])}")
                
                # Safely get detailed analysis
                if 'detailed_analysis' in founder.index and isinstance(founder['detailed_analysis'], str):
                    info.append(f"Analysis: {clean_text_field(founder['detailed_analysis'])}")
                
                founder_info.append("\n".join(info))
            
            founder_context = "\nRelevant Founders:\n" + "\n\n".join(founder_info)
        else:
            founder_context = "\nNo relevant founders found."
        
        # Build prompt based on available data
        prompt = f"""You are a helpful assistant with access to a database of companies and founders.
Query: {query}

Context:
{company_context}
{founder_context}

Previous conversation:
{chr(10).join([f"{'User: ' if msg['is_user'] else 'Assistant: '}{msg['content']}" for msg in conversation_history[-3:]])}

Instructions:
1. Analyze the available data and provide relevant matches for the query
2. For each match, explain why it's relevant
3. Include website and LinkedIn URLs when available
4. Remind users they can request a "deep dive analysis" on any mentioned company or founder
5. If the query asks about company types (like B2B):
   - Analyze the company's products/services from their website content
   - Look for business-focused language in their descriptions
   - Consider their target market based on their content
   - Check their detailed analysis for business model information
6. Format the response in a clear, organized way

Note: If no exact matches are found, analyze the available content to suggest the most relevant options based on the query context."""

        try:
            # Initialize OpenAI client
            client = openai.OpenAI(api_key=st.secrets["openai_api_key"])
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1500
            )
            response_text = response.choices[0].message.content
            
            # Store query and response
            store_query(
                query, 
                response_text,
                num_companies=len(relevant_companies) if not relevant_companies.empty else 0,
                num_founders=len(relevant_founders) if not relevant_founders.empty else 0
            )
            
            return response_text
        except Exception as e:
            error_response = f"Error processing search: {str(e)}"
            store_query(query, error_response, num_companies=0, num_founders=0)
            return error_response

    except Exception as e:
        error_msg = f"Error in query_data: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        return f"An error occurred while processing your query. This has been logged for investigation."

def main():
    # Initialize database
    init_db()
    
    st.set_page_config(page_title="Create-X Search", layout="wide")
    
    # Track visitor
    track_visitor()
    
    # Custom CSS for styling
    st.markdown("""
        <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .stChatMessage {
            background-color: white;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
        }
        .stChatMessage.user {
            background-color: #e3f2fd;
        }
        .stChatMessage.assistant {
            background-color: white;
        }
        .example-queries {
            font-size: 0.9em;
            color: #666;
            margin: 10px 0;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
        }
        .visitor-stats {
            font-size: 0.85em;
            color: #444;
            position: fixed;
            top: 10px;
            right: 10px;
            background-color: rgba(255, 255, 255, 0.95);
            padding: 8px 12px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            z-index: 1000;
            border: 1px solid #eee;
            line-height: 1.6;
        }
        .visitor-stats span {
            display: inline-block;
            margin-right: 12px;
            white-space: nowrap;
        }
        .visitor-stats span:last-child {
            margin-right: 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Page header with logo and stats
    col1, col2, col3 = st.columns([1, 3, 2])
    
    with col1:
        st.image("https://images.crunchbase.com/image/upload/c_pad,h_170,w_170,f_auto,b_white,q_auto:eco,dpr_1/f97zxlq9gei4veywsj1j", width=100)
    
    with col2:
        st.title("Create-X Company and Founder Search for Summer 24")
        st.write("Chat with me to search companies and founders. You can also ask for deep dive analysis on any company or founder!")
    
    with col3:
        stats = get_visitor_stats()
        if stats:
            st.markdown("""
                <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    <h3 style="color: #1f77b4; margin-bottom: 10px; font-size: 1.2em;">Live Stats 📊</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                        <div>
                            <div style="font-weight: bold; color: #2196F3;">Today's Visitors</div>
                            <div style="font-size: 1.5em; color: #1f77b4;">{}</div>
                        </div>
                        <div>
                            <div style="font-weight: bold; color: #2196F3;">Total Visitors</div>
                            <div style="font-size: 1.5em; color: #1f77b4;">{}</div>
                        </div>
                        <div>
                            <div style="font-weight: bold; color: #2196F3;">Total Visits</div>
                            <div style="font-size: 1.5em; color: #1f77b4;">{}</div>
                        </div>
                        <div>
                            <div style="font-weight: bold; color: #2196F3;">Total Queries</div>
                            <div style="font-size: 1.5em; color: #1f77b4;">{}</div>
                        </div>
                    </div>
                </div>
            """.format(
                stats['today_visitors'],
                stats['total_visitors'],
                stats['total_visits'],
                stats['total_queries']
            ), unsafe_allow_html=True)
    
    # Example queries section
    with st.expander("📝 Example Queries - Click to expand"):
        st.markdown("""
        Try asking questions like:
        - Show me companies in the healthcare industry
        - Find founders with experience in artificial intelligence
        - Tell me about companies working on sustainability
        - Who are the founders with software development background?
        - Deep dive analysis of [Company Name]
        - Tell me more about [Founder Name]
        - What companies are focused on B2B solutions?
        - Find founders who studied at Georgia Tech
        """)
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Load data
    companies_df, founders_df = load_data()
    if companies_df is None or founders_df is None:
        st.error("Failed to load data")
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message("user" if message["is_user"] else "assistant"):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about companies or founders..."):
        # Update query count
        update_query_count()
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"is_user": True, "content": prompt})
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                response = query_data(prompt, companies_df, founders_df, st.session_state.messages)
                st.markdown(response)
        st.session_state.messages.append({"is_user": False, "content": response})
    
    # Add a clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

if __name__ == "__main__":
    main() 