from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import pandas as pd
import time
import argparse

def scrape_survivor_pool(url):
    # Setup Chrome options
    options = webdriver.ChromeOptions()
    
    # Initialize the driver
    driver = webdriver.Chrome(options=options)
    driver.maximize_window()
    
    try:
        # Load the page
        print("Loading page...")
        driver.get(url)
        
        # Pause for manual login
        input("Please log in manually in the browser window, then press Enter to continue scraping...")
        
        print("Resuming scraping...")
        time.sleep(5)
        
        # Print page structure to help debug
        print("\nAnalyzing page structure...")
        
        # Try to find the container with all players using Mantine classes
        selectors = [
            "//a[contains(@class, 'mantine-skdmpx')]",  # Main entry class
            "//a[contains(@class, 'isInteractive')]",   # Alternative class
        ]
        
        all_data = []
        last_height = driver.execute_script("return document.body.scrollHeight")
        scroll_attempts = 0
        max_scroll_attempts = 10
        
        print("\nStarting to scroll and collect data...")
        
        # First capture the initial entries before scrolling
        for selector in selectors:
            try:
                initial_entries = driver.find_elements(By.XPATH, selector)
                if initial_entries:
                    print(f"\nFound {len(initial_entries)} initial elements with selector: {selector}")
                    # Process initial entries
                    for entry in initial_entries:
                        try:
                            # Get entry URL to extract player info
                            entry_url = entry.get_attribute('href')
                            entry_id = entry_url.split('entryId=')[1].split('&')[0] if 'entryId=' in entry_url else ''
                            username = entry_url.split('entryUsername=')[1] if 'entryUsername=' in entry_url else ''
                            
                            # Get status based on classes
                            classes = entry.get_attribute('class')
                            status = "Eliminated" if 'lost' in classes else "Active"
                            
                            # Get team pick and additional info from text content
                            text_elements = entry.find_elements(By.XPATH, ".//span[contains(@class, 'mantine-Text-root')]")
                            text_contents = [elem.text for elem in text_elements if elem.text]
                            
                            # First text element is the team pick, rest is additional info
                            team_name = text_contents[0] if text_contents else ""
                            additional_text = ' | '.join(text_contents[1:]) if len(text_contents) > 1 else ""
                            
                            # Create entry data
                            data = {
                                'Username': username,
                                'EntryID': entry_id,
                                'Status': status,
                                'Current_Pick': team_name,
                                'Additional_Info': additional_text
                            }
                            
                            if data not in all_data:
                                all_data.append(data)
                        except Exception as e:
                            continue
                    break
            except Exception as e:
                continue
        
        while scroll_attempts < max_scroll_attempts:
            # Scroll down
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            
            # Try to find player entries
            entries = []
            for selector in selectors:
                try:
                    found_entries = driver.find_elements(By.XPATH, selector)
                    if found_entries:
                        print(f"\nFound {len(found_entries)} elements with selector: {selector}")
                        entries = found_entries
                        break
                except Exception as e:
                    continue
            
            for entry in entries:
                try:
                    # Get entry URL to extract player info
                    entry_url = entry.get_attribute('href')
                    entry_id = entry_url.split('entryId=')[1].split('&')[0] if 'entryId=' in entry_url else ''
                    username = entry_url.split('entryUsername=')[1] if 'entryUsername=' in entry_url else ''
                    
                    # Get status based on classes
                    classes = entry.get_attribute('class')
                    status = "Eliminated" if 'lost' in classes else "Active"
                    
                    # Get team pick and additional info from text content
                    text_elements = entry.find_elements(By.XPATH, ".//span[contains(@class, 'mantine-Text-root')]")
                    text_contents = [elem.text for elem in text_elements if elem.text]
                    
                    # First text element is the team pick, rest is additional info
                    team_name = text_contents[0] if text_contents else ""
                    additional_text = ' | '.join(text_contents[1:]) if len(text_contents) > 1 else ""
                    
                    # Create entry data
                    data = {
                        'Username': username,
                        'EntryID': entry_id,
                        'Status': status,
                        'Current_Pick': team_name,
                        'Additional_Info': additional_text
                    }
                    
                    if data not in all_data:
                        all_data.append(data)
                except Exception as e:
                    # Skip printing individual entry errors
                    continue
            
            # Check if we've reached the bottom
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                scroll_attempts += 1
                print(f"No new content loaded, attempt {scroll_attempts} of {max_scroll_attempts}")
            else:
                scroll_attempts = 0
                last_height = new_height
                print("Found new content, continuing to scroll...")
        
        if all_data:
            # Convert to DataFrame and save to CSV
            df = pd.DataFrame(all_data)
            # Replace NaN values with empty strings
            df = df.fillna('')
            df.to_csv('survivor_pool.csv', index=False)
            print(f"\nSuccessfully scraped {len(df)} entries to survivor_pool.csv")
            print("\nData preview:")
            print(df.head())
        else:
            print("\nNo data was collected. Let's analyze what we can see:")
            print("\nPage source preview (first 1000 characters):")
            print(driver.page_source[:1000])
            print("\nPlease check:")
            print("1. Are you successfully logged in?")
            print("2. Can you see the list of players and their picks?")
            print("3. Right-click on a player entry and select 'Inspect' to see its HTML structure")
            print("4. Share the HTML structure of a single player entry so we can update the selectors")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        input("Press Enter to close the browser...")
        driver.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape survivor pool data from Splash Sports')
    parser.add_argument('url', help='The URL of the survivor pool contest')
    args = parser.parse_args()
    
    scrape_survivor_pool(args.url) 