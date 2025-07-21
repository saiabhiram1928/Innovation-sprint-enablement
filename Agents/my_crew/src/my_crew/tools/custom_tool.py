from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import requests

load_dotenv({})

class NewsSearchToolInput(BaseModel):
    """Input schema for NewsSearchTool."""
    query: str = Field(..., description="Description of the argument.")
    limit : int = Field(10, description="Number of results to return.")


class NewsSearchTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
         "Search for recent news articles on any topic. "
    "Returns title, URL, description, and publish date for each article."
    )
    args_schema: Type[BaseModel] = NewsSearchToolInput

    def _run(self, query: str , limit: int) -> str:
        # Implementation goes here
        try: 
            url = f"https://newsapi.org/v2/everything"
            params = {
                'q' : query,
                'pageSize' : limit,
                'apiKey' : os.getenv('NEWS_API_KEY')
            }
            res = requests.get(url, params=params)
            data = res.json().get('articles', [])
            if res.status_code == 200  and data: 
                articles = []
                for article in data:
                    articles.append({
                        'title': article.get('title'),
                        'description': article.get('description'),
                        'url': article.get('url'),
                        'publishedAt': article.get('publishedAt')
                    })
                return articles
            else:
                return f"Error fetching news: {res.status_code} - {res.text}"
        except Exception as e:
            return f"An error occurred: {str(e)}"
