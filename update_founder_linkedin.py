import os
import openai
import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import random

# Set API keys and credentials from Streamlit secrets
LINKEDIN_USERNAME = st.secrets["linkedin_username"]
LINKEDIN_PASSWORD = st.secrets["linkedin_password"]

# Initialize OpenAI client
client = openai.OpenAI(api_key=st.secrets["openai_api_key"])

def setup_driver():
    """Setup Chrome driver with appropriate options"""
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('--disable-notifications')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver

def linkedin_login(driver):
    """Login to LinkedIn"""
    try:
        driver.get('https://www.linkedin.com/login')
        time.sleep(3)
        
        username = driver.find_element(By.ID, 'username')
        password = driver.find_element(By.ID, 'password')
        
        username.send_keys(LINKEDIN_USERNAME)
        password.send_keys(LINKEDIN_PASSWORD)
        
        driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
        time.sleep(5)
        
        if 'feed' in driver.current_url:
            print("Successfully logged in to LinkedIn")
            return True
        else:
            print("Login might have failed. Current URL:", driver.current_url)
            return False
            
    except Exception as e:
        print(f"Error during login: {str(e)}")
        return False

def extract_linkedin_data(driver, linkedin_url):
    """Extract profile information using Selenium"""
    try:
        driver.get(linkedin_url)
        time.sleep(3)
        
        profile_data = {
            'headline': '',
            'summary': '',
            'education': [],
            'experience': [],
            'skills': [],
            'certifications': [],
            'languages': [],
            'volunteer': [],
            'posts': []
        }
        
        # Extract headline
        try:
            headline = driver.find_element(By.CSS_SELECTOR, "div.text-body-medium").text
            profile_data['headline'] = headline
        except:
            pass
            
        # Extract summary/about
        try:
            summary = driver.find_element(By.CSS_SELECTOR, "div.pv-shared-text-with-see-more").text
            profile_data['summary'] = summary
        except:
            pass
        
        # Extract education
        education_section = driver.find_elements(By.CSS_SELECTOR, "section.education-section li")
        for edu in education_section:
            try:
                profile_data['education'].append(edu.text)
            except:
                continue
        
        # Extract experience
        experience_section = driver.find_elements(By.CSS_SELECTOR, "section.experience-section li")
        for exp in experience_section:
            try:
                profile_data['experience'].append(exp.text)
            except:
                continue
        
        # Extract skills
        try:
            skills_button = driver.find_element(By.CSS_SELECTOR, "button[aria-label='Skills']")
            skills_button.click()
            time.sleep(2)
            skills = driver.find_elements(By.CSS_SELECTOR, "span.skill-name")
            profile_data['skills'] = [skill.text for skill in skills]
        except:
            pass
        
        # Extract certifications
        try:
            certifications = driver.find_elements(By.CSS_SELECTOR, "section.certifications-section li")
            profile_data['certifications'] = [cert.text for cert in certifications]
        except:
            pass
            
        # Extract languages
        try:
            languages = driver.find_elements(By.CSS_SELECTOR, "section.languages-section li")
            profile_data['languages'] = [lang.text for lang in languages]
        except:
            pass
            
        # Extract volunteer experience
        try:
            volunteer = driver.find_elements(By.CSS_SELECTOR, "section.volunteer-section li")
            profile_data['volunteer'] = [vol.text for vol in volunteer]
        except:
            pass
            
        # Extract recent posts
        try:
            print("Attempting to extract posts...")
            
            if linkedin_url:
                activity_url = linkedin_url + "recent-activity/shares/"
                driver.get(activity_url)
                time.sleep(5)
                print("Navigated to activity page")
            
            post_selectors = [
                "div.update-components-actor",
                "div.profile-creator-shared-feed-update",
                "div.feed-shared-update-v2__description-wrapper",
                "div.update-components-text"
            ]
            
            posts = []
            for selector in post_selectors:
                posts = driver.find_elements(By.CSS_SELECTOR, selector)
                if posts:
                    print(f"Found {len(posts)} posts using selector: {selector}")
                    break
            
            if not posts:
                print("No posts found initially, trying to scroll...")
                for _ in range(3):
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)
                    for selector in post_selectors:
                        posts = driver.find_elements(By.CSS_SELECTOR, selector)
                        if posts:
                            print(f"Found {len(posts)} posts after scrolling using selector: {selector}")
                            break
                    if posts:
                        break
            
            post_data = []
            for post in posts[:10]:
                try:
                    text_selectors = [
                        "div.update-components-text",
                        "span.break-words",
                        "div.feed-shared-update-v2__description-wrapper",
                        "div.update-components-text--native-line-height"
                    ]
                    
                    is_repost = False
                    repost_indicators = [
                        "div.update-components-header",
                        "span.update-components-header__text",
                        "div.update-components-actor--with-supplementary-actor-info"
                    ]
                    
                    for indicator in repost_indicators:
                        try:
                            repost_element = post.find_element(By.CSS_SELECTOR, indicator)
                            if repost_element and any(word in repost_element.text.lower() for word in ['reposted', 'shared']):
                                is_repost = True
                                try:
                                    original_author = post.find_element(By.CSS_SELECTOR, "span.update-components-actor__supplementary-actor-info").text
                                except:
                                    original_author = "Unknown"
                                break
                        except:
                            continue
                    
                    post_text = ""
                    for text_selector in text_selectors:
                        try:
                            text_elements = post.find_elements(By.CSS_SELECTOR, text_selector)
                            if text_elements:
                                post_text = " ".join([elem.text for elem in text_elements if elem.text])
                                if post_text:
                                    break
                        except:
                            continue
                    
                    repost_comment = ""
                    if is_repost:
                        comment_selectors = [
                            "div.update-components-text--repost",
                            "div.update-components-commentary",
                            "div.feed-shared-update-v2__commentary"
                        ]
                        for comment_selector in comment_selectors:
                            try:
                                comment_elements = post.find_elements(By.CSS_SELECTOR, comment_selector)
                                if comment_elements:
                                    repost_comment = " ".join([elem.text for elem in comment_elements if elem.text])
                                    if repost_comment:
                                        break
                            except:
                                continue
                    
                    date_selectors = [
                        "span.update-components-actor__sub-description",
                        "time.update-components-actor__sub-description",
                        "span.update-components-text-view time-badge"
                    ]
                    
                    post_date = ""
                    for date_selector in date_selectors:
                        try:
                            date_element = post.find_element(By.CSS_SELECTOR, date_selector)
                            if date_element:
                                post_date = date_element.text
                                break
                        except:
                            continue
                    
                    reaction_selectors = [
                        "span.social-details-social-counts__reactions-count",
                        "button.social-details-social-counts__count-value",
                        "span.update-v2-social-activity"
                    ]
                    
                    reaction_count = "0"
                    for reaction_selector in reaction_selectors:
                        try:
                            reactions = post.find_elements(By.CSS_SELECTOR, reaction_selector)
                            if reactions:
                                reaction_count = reactions[0].text
                                break
                        except:
                            continue
                    
                    if post_text or post_date:
                        post_info = {
                            'text': post_text.strip(),
                            'date': post_date.strip(),
                            'reactions': reaction_count.strip(),
                            'is_repost': is_repost,
                            'original_author': original_author if is_repost else None,
                            'repost_comment': repost_comment.strip() if is_repost else None
                        }
                        post_data.append(post_info)
                        if is_repost:
                            print(f"Extracted repost from {post_date} (Original: {original_author})")
                            if repost_comment:
                                print(f"With comment: {repost_comment[:100]}...")
                        else:
                            print(f"Extracted original post from {post_date}: {post_text[:100]}...")
                except Exception as e:
                    print(f"Error extracting individual post: {e}")
                    continue
            
            profile_data['posts'] = post_data
            print(f"Successfully extracted {len(post_data)} posts")
            
        except Exception as e:
            print(f"Error extracting posts: {e}")
        
        return profile_data
    except Exception as e:
        print(f"Error extracting data: {e}")
        return None

def analyze_founder_background(profile_data):
    """Analyze founder background using GPT-4o-mini"""
    if not profile_data:
        return "No profile data available for analysis."
    
    prompt = f"""
    Analyze this founder's background based on their LinkedIn profile:
    
    Headline: {profile_data.get('headline', 'N/A')}
    Summary: {profile_data.get('summary', 'N/A')}
    Experience: {', '.join(profile_data.get('experience', ['N/A']))}
    Education: {', '.join(profile_data.get('education', ['N/A']))}
    Skills: {', '.join(profile_data.get('skills', ['N/A']))}
    Certifications: {', '.join(profile_data.get('certifications', ['N/A']))}
    Languages: {', '.join(profile_data.get('languages', ['N/A']))}
    Volunteer Experience: {', '.join(profile_data.get('volunteer', ['N/A']))}
    
    Recent Posts:
    {chr(10).join([f"- {post.get('text', '')} ({post.get('date', '')})" for post in profile_data.get('posts', [])[:3]])}
    
    Please provide a comprehensive analysis covering:
    1. Educational background and qualifications
    2. Career progression and key roles
    3. Areas of expertise and skills
    4. Industry experience
    5. Notable achievements
    6. Entrepreneurial experience
    7. Technical vs. business focus
    8. Leadership experience
    9. Communication style and thought leadership (based on posts)
    10. Areas of interest and industry involvement
    """
    
    try:
        # Using the global client initialized above
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return None

def is_valid_linkedin_url(url):
    """Validate if the URL is a legitimate LinkedIn profile URL"""
    if not url:
        return False
    url = url.lower().strip()
    return (
        url.startswith('https://www.linkedin.com/') or 
        url.startswith('http://www.linkedin.com/') or
        url.startswith('www.linkedin.com/') or
        url.startswith('linkedin.com/')
    )

def main():
    # Load both datasets
    try:
        original_df = pd.read_csv('createx_founders.csv')
        enriched_df = pd.read_pickle('enriched_founders.pkl')
        print("Successfully loaded both datasets")
        print("Original columns:", original_df.columns.tolist())
        print("Enriched columns:", enriched_df.columns.tolist())
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Find founders with invalid or missing LinkedIn URLs
    missing_linkedin = enriched_df[
        (enriched_df['linkedin_url'].isna()) | 
        (enriched_df['linkedin_url'] == '') | 
        (enriched_df['linkedin_url'] == 'N/A')
    ]
    
    # Also check original CSV for invalid LinkedIn URLs
    for idx, row in original_df.iterrows():
        linkedin = str(row.get('linkedin', '')).strip()
        if linkedin and not is_valid_linkedin_url(linkedin):
            print(f"\nFound invalid LinkedIn URL in original CSV: {linkedin} for {row['name']}")
            # Find corresponding row in enriched_df
            enriched_idx = enriched_df[enriched_df['Name'] == row['name']].index
            if len(enriched_idx) > 0:
                # Add to missing_linkedin if not already there
                if enriched_idx[0] not in missing_linkedin.index:
                    missing_linkedin = pd.concat([missing_linkedin, enriched_df.loc[enriched_idx]])
    
    if len(missing_linkedin) == 0:
        print("No founders missing LinkedIn URLs!")
        return
    
    print(f"\nFound {len(missing_linkedin)} founders missing LinkedIn URLs")
    
    # Setup Chrome driver
    driver = setup_driver()
    
    try:
        # Login to LinkedIn
        if not linkedin_login(driver):
            print("Failed to login to LinkedIn")
            driver.quit()
            return
        
        # Process each founder missing LinkedIn
        for idx, founder in missing_linkedin.iterrows():
            current_linkedin = None
            # Check if there's an invalid URL in original CSV
            original_idx = original_df[original_df['name'] == founder['Name']].index
            if len(original_idx) > 0:
                current_linkedin = str(original_df.at[original_idx[0], 'linkedin']).strip()
            
            print(f"\nProcessing: {founder['Name']} from {founder.get('Company', 'Unknown Company')}")
            if current_linkedin and not is_valid_linkedin_url(current_linkedin):
                print(f"Current invalid LinkedIn URL: {current_linkedin}")
            
            # Ask for LinkedIn URL
            print("Options:")
            print("1. Enter LinkedIn URL")
            print("2. Type 'NA' if no LinkedIn profile exists")
            print("3. Press Enter to skip this founder")
            
            linkedin_url = input(f"Please choose an option for {founder['Name']}: ").strip()
            
            if not linkedin_url:
                print("Skipping this founder...")
                continue
            
            if linkedin_url.upper() == 'NA':
                print(f"Marking {founder['Name']} as having no LinkedIn profile")
                # Update enriched DataFrame
                enriched_df.at[idx, 'linkedin_url'] = 'N/A'
                enriched_df.at[idx, 'detailed_analysis'] = 'No LinkedIn profile available.'
                
                # Update original DataFrame
                if len(original_idx) > 0:
                    original_df.at[original_idx[0], 'linkedin'] = 'N/A'
                
                # Save progress
                enriched_df.to_pickle('enriched_founders.pkl')
                enriched_df.to_csv('enriched_founders.csv', index=False)
                original_df.to_csv('createx_founders.csv', index=False)
                continue
            
            # Validate LinkedIn URL
            if not is_valid_linkedin_url(linkedin_url):
                print("Invalid LinkedIn URL. URL must start with 'linkedin.com' or 'www.linkedin.com'")
                continue
            
            # Ensure URL starts with https://
            if not linkedin_url.startswith('http'):
                linkedin_url = 'https://' + linkedin_url.lstrip('/')
            
            # Extract and analyze LinkedIn data
            profile_data = extract_linkedin_data(driver, linkedin_url)
            if profile_data:
                analysis = analyze_founder_background(profile_data)
                
                # Update enriched DataFrame
                enriched_df.at[idx, 'linkedin_url'] = linkedin_url
                enriched_df.at[idx, 'detailed_analysis'] = analysis
                enriched_df.at[idx, 'linkedin_headline'] = profile_data.get('headline', '')
                enriched_df.at[idx, 'linkedin_summary'] = profile_data.get('summary', '')
                enriched_df.at[idx, 'linkedin_experience'] = profile_data.get('experience', [])
                enriched_df.at[idx, 'linkedin_education'] = profile_data.get('education', [])
                enriched_df.at[idx, 'linkedin_skills'] = profile_data.get('skills', [])
                enriched_df.at[idx, 'linkedin_certifications'] = profile_data.get('certifications', [])
                enriched_df.at[idx, 'linkedin_languages'] = profile_data.get('languages', [])
                enriched_df.at[idx, 'linkedin_volunteer'] = profile_data.get('volunteer', [])
                enriched_df.at[idx, 'linkedin_posts'] = profile_data.get('posts', [])
                
                # Update original DataFrame
                if len(original_idx) > 0:
                    original_df.at[original_idx[0], 'linkedin'] = linkedin_url
                
                # Save progress
                enriched_df.to_pickle('enriched_founders.pkl')
                enriched_df.to_csv('enriched_founders.csv', index=False)
                original_df.to_csv('createx_founders.csv', index=False)
                
                print(f"Updated and saved LinkedIn info for {founder['Name']}")
            else:
                print(f"Failed to extract LinkedIn data for {founder['Name']}")
            
            time.sleep(3)  # Delay between requests
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    
    finally:
        driver.quit()
        print("\nScript completed")

if __name__ == "__main__":
    main() 