# analyze_html_response.py
from bs4 import BeautifulSoup
import re
import json

def analyze_html_response():
    """Analyze the HTML response to understand what we're getting"""
    try:
        with open('api_response.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        print("🔍 Analyzing HTML Response...")
        print(f"Title: {soup.title.string if soup.title else 'No title'}")
        
        # Look for forms (login forms)
        forms = soup.find_all('form')
        print(f"\n📝 Found {len(forms)} form(s):")
        for form in forms:
            action = form.get('action', 'No action')
            print(f"  - Form action: {action}")
        
        # Look for JavaScript variables that might contain data
        scripts = soup.find_all('script')
        print(f"\n📜 Found {len(scripts)} script(s)")
        
        # Look for CDR-related data in scripts
        for script in scripts:
            if script.string:
                # Look for JSON data or CDR references
                if 'cdr' in script.string.lower() or 'call' in script.string.lower():
                    print("  - Script contains CDR/call references")
                    # Try to extract JSON-like data
                    json_matches = re.findall(r'\{[^{}]*"[^"]*"[^{}]*\}', script.string)
                    if json_matches:
                        print("  - Found potential JSON data snippets")
        
        # Look for API endpoints in the page
        links = soup.find_all('a', href=True)
        api_links = [link['href'] for link in links if 'api' in link['href'].lower()]
        if api_links:
            print(f"\n🔗 Found API-related links:")
            for link in api_links[:5]:  # Show first 5
                print(f"  - {link}")
        
        print("\n💡 Analysis complete.")
        
    except FileNotFoundError:
        print("❌ api_response.html not found. Run the main script first.")

if __name__ == "__main__":
    analyze_html_response()
    