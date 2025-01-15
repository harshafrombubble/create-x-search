import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import random

def get_headers():
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }

def get_founder_info(company_url):
    try:
        time.sleep(random.uniform(1, 2))
        
        response = requests.get(company_url, headers=get_headers())
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        founder_info = []
        
        # Find the team members view
        team_view = soup.find('div', class_='view-cx-team-members')
        if not team_view:
            print(f"No team members found for {company_url}")
            return [], None, None, None, None
            
        # Find all team member items
        team_members = team_view.find_all('div', class_='entity__team-member--teaser')
        
        for member in team_members:
            # Get member name
            name_field = member.find('div', class_='field--name-field-cx-member-full-name')
            if not name_field:
                continue
                
            name = name_field.get_text(strip=True)
            
            # Get member role/major
            role = 'Founder'  # Default role
            major_field = member.find('div', class_='field--name-field-cx-team-undergraduate-majo')
            if major_field:
                role = major_field.get_text(strip=True)
            
            # Get LinkedIn URL
            linkedin = None
            linkedin_link = member.find('a', class_='button__georgiatech--mono')
            if linkedin_link:
                linkedin = linkedin_link.get('href')
            
            founder_data = {
                'name': name,
                'role': role,
                'linkedin': linkedin
            }
            
            print(f"Found founder: {name}")
            founder_info.append(founder_data)
        
        # Get company details
        company_email = None
        company_industry = None
        company_market = None
        company_website = None
        
        # Try to get company website
        website_link = soup.find('a', string='Website')
        if website_link:
            company_website = website_link.get('href')
        
        # Try to get company email
        email_field = soup.find('h2', string='Company Email Address')
        if email_field and email_field.find_next('p'):
            company_email = email_field.find_next('p').get_text(strip=True)
        
        # Try to get industry
        industry_field = soup.find('h2', string='Industry')
        if industry_field and industry_field.find_next('p'):
            company_industry = industry_field.find_next('p').get_text(strip=True)
        
        # Try to get market segment
        market_field = soup.find('h2', string='Market Segment')
        if market_field and market_field.find_next('p'):
            company_market = market_field.find_next('p').get_text(strip=True)
        
        print(f"Total founders found: {len(founder_info)}")
        return founder_info, company_email, company_industry, company_market, company_website
        
    except Exception as e:
        print(f"Error fetching founder info from {company_url}: {e}")
        print(f"Full error: {str(e)}")
        return [], None, None, None, None

def scrape_createx_demoday():
    base_url = 'https://create-x.gatech.edu'
    url = f'{base_url}/demoday'
    
    try:
        response = requests.get(url, headers=get_headers())
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        companies = []
        founders = []
        
        # Find the main table
        table = soup.find('table')
        if not table:
            raise Exception("Could not find company table")
            
        rows = table.find_all('tr')
        
        for row in rows[1:]:  # Skip header row
            cols = row.find_all('td')
            if len(cols) >= 4:
                company_link = cols[0].find('a')
                company_url = base_url + company_link['href'] if company_link else None
                company_name = cols[0].get_text(strip=True)
                
                # Extract video URL from column 1
                video_link = cols[1].find('a')
                video_url = video_link['href'] if video_link else 'N/A'
                
                if company_url:
                    print(f"Fetching founder information for {company_name}...")
                    founder_list, company_email, company_industry, company_market, company_website = get_founder_info(company_url)
                    
                    company_data = {
                        'Company': company_name,
                        'Video_URL': video_url,
                        'Description': cols[2].get_text(strip=True) if len(cols) > 2 else '',
                        'Industry': company_industry or (cols[3].get_text(strip=True) if len(cols) > 3 else ''),
                        'Market_Segment': company_market or '',
                        'Email': company_email or '',
                        'Website': company_website or '',
                        'Booth': cols[4].get_text(strip=True) if len(cols) > 4 else 'N/A',
                        'Company_URL': company_url
                    }
                    companies.append(company_data)
                    
                    for founder in founder_list:
                        founder['company'] = company_name
                        founders.append(founder)
        
        # Create DataFrames
        companies_df = pd.DataFrame(companies)
        
        # Handle empty founders list
        if not founders:
            founders_df = pd.DataFrame(columns=['name', 'company', 'role', 'linkedin'])
        else:
            founders_df = pd.DataFrame(founders)
        
        # Ensure columns are in the right order
        company_columns = ['Company', 'Video_URL', 'Description', 'Industry', 'Market_Segment', 'Email', 'Website', 'Booth', 'Company_URL']
        founder_columns = ['name', 'company', 'role', 'linkedin']
        
        companies_df = companies_df[company_columns]
        founders_df = founders_df[founder_columns]
        
        # Save to CSV with proper encoding
        companies_df.to_csv('createx_companies.csv', index=False, encoding='utf-8')
        founders_df.to_csv('createx_founders.csv', index=False, encoding='utf-8')
        
        print(f"Successfully scraped {len(companies)} companies and {len(founders)} founders!")
        print("Data saved to CSV files")
        
        return companies_df, founders_df
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

if __name__ == "__main__":
    scrape_createx_demoday() 