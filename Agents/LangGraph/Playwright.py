import nest_asyncio
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser  
import textwrap
import asyncio
from playwright.sync_api import sync_playwright

nest_asyncio.apply()
async_browser =  create_async_playwright_browser(headless=False)  
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = toolkit.get_tools()
async def tool_testing():
    tool_dict = {tool.name: tool for tool in tools}
    navigate_tool = tool_dict.get("navigate_browser")
    extract_text_tool = tool_dict.get("extract_text")
    
    print("Navigating to CNN...")
    await navigate_tool.arun({"url": "https://www.cnn.com"})
    
    print("Extracting text...")
    text = await extract_text_tool.arun({}) 
    wrapped_text = textwrap.fill(text[:500])  # Show first 500 chars
    print("Extracted text:")
    print(wrapped_text)
    return wrapped_text
def test_sync_playwright():
    # Use sync Playwright for more reliable execution
    p = sync_playwright().start()
    browser = p.chromium.launch(headless=False)
    toolkit = PlayWrightBrowserToolkit.from_browser(browser)
    tools = toolkit.get_tools()
    
    tool_dict = {tool.name: tool for tool in tools}
    navigate_tool = tool_dict.get("navigate_browser")
    extract_text_tool = tool_dict.get("extract_text")
    
    try:
        print("Navigating to Google...")
        # Use a more reliable website
        navigate_tool.run({"url": "https://www.google.com"})
        
        print("Navigation successful! Extracting text...")
        text = extract_text_tool.run({})
        wrapped_text = textwrap.fill(text[:500])
        print("Extracted text:")
        print(wrapped_text)
        
        # Clean up
        browser.close()
        p.stop()
        return wrapped_text
    except Exception as e:
        print(f"Error: {e}")
        browser.close()
        p.stop()
        return None
# asyncio.run(tool_testing())
test_sync_playwright()