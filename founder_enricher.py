import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import openai
import os
import streamlit as st
import logging

# Set API keys and credentials from Streamlit secrets
openai.api_key = st.secrets["openai_api_key"]
LINKEDIN_USERNAME = st.secrets["linkedin_username"]
LINKEDIN_PASSWORD = st.secrets["linkedin_password"]

def setup_driver():
    """Setup Chrome driver with appropriate options"""
    chrome_options = Options()
    # chrome_options.add_argument('--headless')  # Commenting out headless mode
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('--disable-notifications')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Use webdriver_manager to handle driver installation
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Execute CDP commands to prevent detection
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver

def linkedin_login(driver):
    """Login to LinkedIn using credentials from Streamlit secrets"""
    try:
        driver.get("https://www.linkedin.com/login")
        time.sleep(2)
        
        # Get credentials from Streamlit secrets with error handling
        try:
            username = st.secrets["linkedin_username"]
            password = st.secrets["linkedin_password"]
        except Exception as e:
            logging.error(f"Error loading LinkedIn credentials from secrets: {str(e)}")
            st.error("""Error loading LinkedIn credentials. Please ensure your .streamlit/secrets.toml file contains:
            1. linkedin_username
            2. linkedin_password
            And check for proper formatting of special characters.""")
            return False
        
        # Find and fill username
        username_field = driver.find_element(By.ID, "username")
        username_field.send_keys(username)
        
        # Find and fill password
        password_field = driver.find_element(By.ID, "password")
        password_field.send_keys(password)
        
        # Click login button
        login_button = driver.find_element(By.CSS_SELECTOR, "[type='submit']")
        login_button.click()
        
        time.sleep(3)  # Wait for login to complete
        
        return True
    except Exception as e:
        logging.error(f"LinkedIn login failed: {str(e)}")
        return False

def extract_linkedin_data(driver, linkedin_url):
    """Extract profile information using Selenium"""
    try:
        driver.get(linkedin_url)
        time.sleep(3)  # Wait for page to load
        
        # Extract basic information
        profile_data = {
            'headline': '',
            'summary': '',
            'education': [],
            'experience': [],
            'skills': [],
            'certifications': [],
            'languages': [],
            'volunteer': [],
            'posts': []  # Added posts field
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
            
            # First try to navigate to the activity page directly if we have the profile URL
            if linkedin_url:
                activity_url = linkedin_url + "recent-activity/shares/"
                driver.get(activity_url)
                time.sleep(5)
                print("Navigated to activity page")
            
            # Try different post container selectors based on actual LinkedIn structure
            post_selectors = [
                "div.update-components-actor",  # Main post container
                "div.profile-creator-shared-feed-update",  # Profile activity feed
                "div.feed-shared-update-v2__description-wrapper",  # Post content wrapper
                "div.update-components-text" # Post text container
            ]
            
            posts = []
            for selector in post_selectors:
                posts = driver.find_elements(By.CSS_SELECTOR, selector)
                if posts:
                    print(f"Found {len(posts)} posts using selector: {selector}")
                    break
            
            # If no posts found, try scrolling
            if not posts:
                print("No posts found initially, trying to scroll...")
                for _ in range(3):  # Try scrolling up to 3 times
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
            for post in posts[:10]:  # Get last 10 posts
                try:
                    # Extract post text using updated selectors
                    text_selectors = [
                        "div.update-components-text",  # Main text container
                        "span.break-words",  # Text content
                        "div.feed-shared-update-v2__description-wrapper",  # Post wrapper
                        "div.update-components-text--native-line-height"  # Text with native line height
                    ]
                    
                    # Determine if this is a repost
                    is_repost = False
                    repost_indicators = [
                        "div.update-components-header",  # Header section that shows repost info
                        "span.update-components-header__text",  # "Reposted" text
                        "div.update-components-actor--with-supplementary-actor-info"  # Shows original poster
                    ]
                    
                    for indicator in repost_indicators:
                        try:
                            repost_element = post.find_element(By.CSS_SELECTOR, indicator)
                            if repost_element and any(word in repost_element.text.lower() for word in ['reposted', 'shared']):
                                is_repost = True
                                # Try to get original author
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
                    
                    # Get any comment on the repost
                    repost_comment = ""
                    if is_repost:
                        comment_selectors = [
                            "div.update-components-text--repost",  # Repost comment section
                            "div.update-components-commentary",    # Commentary on repost
                            "div.feed-shared-update-v2__commentary"  # Another commentary format
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
                    
                    # Extract date using updated selectors
                    date_selectors = [
                        "span.update-components-actor__sub-description",  # Post date
                        "time.update-components-actor__sub-description",  # Time element
                        "span.update-components-text-view time-badge" # Time badge
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
                    
                    # Extract reactions using updated selectors
                    reaction_selectors = [
                        "span.social-details-social-counts__reactions-count",  # Reaction count
                        "button.social-details-social-counts__count-value",  # Reaction button
                        "span.update-v2-social-activity" # New reaction count format
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
                    
                    # Only add post if we found some content
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

def analyze_founder_background(profile_data, max_retries=3):
    """Use GPT to analyze founder's background and skills"""
    if not profile_data:
        return {}
    
    # Format posts for analysis
    posts_text = ""
    if profile_data.get('posts'):
        posts_text = "\n".join([
            f"Post ({post['date']}): {post['text']}\nReactions: {post['reactions']}"
            for post in profile_data['posts']
        ])
    
    prompt = f"""
    Analyze this founder's background based on their LinkedIn profile:
    
    Headline: {profile_data.get('headline', '')}
    Summary: {profile_data.get('summary', '')}
    Education: {profile_data.get('education', [])}
    Experience: {profile_data.get('experience', [])}
    Skills: {profile_data.get('skills', [])}
    Certifications: {profile_data.get('certifications', [])}
    Languages: {profile_data.get('languages', [])}
    Volunteer Experience: {profile_data.get('volunteer', [])}
    
    Recent LinkedIn Activity:
    {posts_text}
    
    Please provide:
    1. Their educational background and major(s)
    2. Their role and responsibilities in their current company
    3. Key skills and expertise
    4. Areas of interest and thought leadership based on their posts
    5. Notable achievements or experience
    6. Communication style and engagement (based on posts)
    7. Professional focus and interests
    8. Industry involvement and network
    """
    
    for attempt in range(max_retries):
        try:
            print(f"Attempting to analyze background (attempt {attempt + 1}/{max_retries})...")
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Using GPT-4o-mini for faster, more affordable analysis
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
                timeout=30  # Add timeout
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error analyzing background (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # Exponential backoff
                print(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Skipping background analysis.")
                return "Analysis failed due to API errors."

def enrich_founder_data():
    """Main function to enrich founder data"""
    # Read founders DataFrame
    try:
        founders_df = pd.read_csv('createx_founders.csv')
        print("Successfully loaded founders data")
    except Exception as e:
        print(f"Error loading founders data: {str(e)}")
        return
    
    # Print column names to debug
    print("Available columns:", founders_df.columns.tolist())
    
    # Rename columns if needed
    column_mapping = {
        'name': 'Name',
        'company': 'Company',
        'linkedin': 'LinkedIn_URL'
    }
    founders_df = founders_df.rename(columns=column_mapping)
    
    # Verify required columns exist
    required_columns = ['Name']
    missing_columns = [col for col in required_columns if col not in founders_df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return
    
    output_file = 'enriched_founders.pkl'
    
    # Check if output file exists and load existing data
    try:
        if os.path.exists(output_file):
            enriched_df = pd.read_pickle(output_file)
            processed_founders = set(enriched_df['Name'])
            print(f"Found existing data for {len(processed_founders)} founders")
        else:
            enriched_df = pd.DataFrame()
            processed_founders = set()
    except Exception as e:
        print(f"Error loading existing data: {str(e)}")
        enriched_df = pd.DataFrame()
        processed_founders = set()
    
    # Setup Chrome driver for LinkedIn
    try:
        driver = setup_driver()
    except Exception as e:
        print(f"Error setting up Chrome driver: {str(e)}")
        return
    
    # Login to LinkedIn
    try:
        if not linkedin_login(driver):
            print("Failed to login to LinkedIn")
            driver.quit()
            return
    except Exception as e:
        print(f"Error during LinkedIn login: {str(e)}")
        driver.quit()
        return
    
    try:
        total_founders = len(founders_df)
        processed_count = 0
        
        for idx, founder in founders_df.iterrows():
            if founder['Name'] in processed_founders:
                print(f"Skipping {founder['Name']} (already processed)")
                continue
                
            print(f"\nProcessing founder {processed_count + 1}/{total_founders}: {founder['Name']}")
            
            # Extract LinkedIn data
            profile_data = None
            linkedin_url = None
            
            # First check if LinkedIn URL is in the data
            if pd.notna(founder.get('LinkedIn_URL')):
                linkedin_url = founder['LinkedIn_URL']
                print(f"Found LinkedIn URL in data: {linkedin_url}")
            
            # If no LinkedIn URL, try searching
            if not linkedin_url:
                try:
                    search_url = f"https://www.linkedin.com/search/results/people/?keywords={founder['Name']}"
                    driver.get(search_url)
                    time.sleep(5)
                    
                    # Look for profile links
                    profile_links = driver.find_elements(By.CSS_SELECTOR, "a.app-aware-link")
                    for link in profile_links:
                        try:
                            href = link.get_attribute('href')
                            if href and '/in/' in href.lower():
                                linkedin_url = href
                                print(f"Found LinkedIn URL: {linkedin_url}")
                                break
                        except Exception as e:
                            print(f"Error getting link href: {str(e)}")
                            continue
                except Exception as e:
                    print(f"Error during LinkedIn search: {str(e)}")
            
            if linkedin_url:
                print(f"Extracting data from LinkedIn: {linkedin_url}")
                try:
                    profile_data = extract_linkedin_data(driver, linkedin_url)
                except Exception as e:
                    print(f"Error extracting LinkedIn data: {str(e)}")
                    profile_data = None
            else:
                print(f"Could not find LinkedIn URL for {founder['Name']}")
            
            # Analyze data
            analysis = analyze_founder_background(profile_data) if profile_data else "No LinkedIn data available for analysis."
            
            # Create a new row of data
            try:
                new_data = {
                    'Name': founder['Name'],
                    'Company': founder.get('Company', ''),
                    'linkedin_url': linkedin_url if linkedin_url else '',
                    'detailed_analysis': analysis,
                    'linkedin_headline': profile_data.get('headline', '') if profile_data else '',
                    'linkedin_summary': profile_data.get('summary', '') if profile_data else '',
                    'linkedin_experience': profile_data.get('experience', []) if profile_data else [],
                    'linkedin_education': profile_data.get('education', []) if profile_data else [],
                    'linkedin_skills': profile_data.get('skills', []) if profile_data else [],
                    'linkedin_certifications': profile_data.get('certifications', []) if profile_data else [],
                    'linkedin_languages': profile_data.get('languages', []) if profile_data else [],
                    'linkedin_volunteer': profile_data.get('volunteer', []) if profile_data else [],
                    'linkedin_posts': profile_data.get('posts', []) if profile_data else []
                }
                
                # Append new data to DataFrame
                new_row_df = pd.DataFrame([new_data])
                enriched_df = pd.concat([enriched_df, new_row_df], ignore_index=True)
                
                # Save progress after each founder
                try:
                    enriched_df.to_pickle(output_file)
                    print(f"Progress saved for {founder['Name']}")
                    
                    # Also save a CSV backup
                    enriched_df.to_csv('enriched_founders.csv', index=False)
                except Exception as e:
                    print(f"Error saving progress: {str(e)}")
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing founder data: {str(e)}")
                continue
            
            # Respect rate limits
            time.sleep(5)
    
    except Exception as e:
        print(f"Error in main processing loop: {str(e)}")
    
    finally:
        try:
            driver.quit()
        except:
            pass
        
        print(f"\nCompleted processing {processed_count} founders")
        print(f"Total founders in enriched data: {len(enriched_df)}")

if __name__ == "__main__":
    enrich_founder_data()
