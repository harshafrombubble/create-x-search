import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

def get_headers():
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }

def scrape_createx_demoday():
    url = 'https://create-x.gatech.edu/demoday'
    companies = []
    
    try:
        # Add a small random delay
        time.sleep(random.uniform(1, 3))
        
        response = requests.get(url, headers=get_headers())
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all table rows
        rows = soup.find_all('tr')
        
        for row in rows:
            if not row.find_all('td'):
                continue
                
            cols = row.find_all('td')
            if len(cols) >= 4:
                # Extract video URL if available
                video_link = cols[1].find('a')['href'] if cols[1].find('a') else 'N/A'
                
                company_data = {
                    'Company': cols[0].get_text(strip=True),
                    'Video_URL': video_link,
                    'Description': cols[2].get_text(strip=True),
                    'Industry': cols[3].get_text(strip=True),
                    'Booth': cols[4].get_text(strip=True) if len(cols) > 4 else 'N/A'
                }
                companies.append(company_data)
        
        df = pd.DataFrame(companies)
        
        # Save to both CSV and Excel for flexibility
        df.to_csv('createx_companies.csv', index=False)
        df.to_excel('createx_companies.xlsx', index=False)
        
        print(f"Successfully scraped {len(companies)} companies!")
        print("Data saved to 'createx_companies.csv' and 'createx_companies.xlsx'")
        
        return df
        
    except requests.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    scrape_createx_demoday() 