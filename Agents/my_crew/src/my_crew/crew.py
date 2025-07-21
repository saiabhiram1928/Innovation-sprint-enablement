from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from typing import List
from my_crew.tools.custom_tool import NewsSearchTool

@CrewBase
class MyCrew():
    """MyCrew crew"""

    agents: List[Agent]
    tasks: List[Task]

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    @agent
    def newsGatherAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['news_gather'], # type: ignore[index]
            verbose=True,
            tools = [NewsSearchTool()],
            vars= {
                'topic': 'Top 10 frameworks for building AI agents',
                'current_year': '2023'
            }
        )

    @agent
    def articleSummarizerAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['article_summarizer'], # type: ignore[index]
            verbose=True
        )
    @agent
    def sentimentAnalyzerAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['sentiment_analyzer'] # type: ignore[index]
        )
    @agent
    def digestFormatterAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['digest_formatter'] # type: ignore[index]
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def research_new_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_new_task'], # type: ignore[index]
            tools = [NewsSearchTool()]
        )

    @task
    def summarize_articles_task(self) -> Task:
        return Task(
            config=self.tasks_config['summarize_articles_task'], # type: ignore[index]
            output_file='report.md'
        )
    @task
    def analyze_sentiment_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_sentiment_task'],
        )
    @task
    def format_digest_task(self) -> Task:
        return Task(
            config=self.tasks_config['format_digest_task'],
            output_file='daily_new_summarize.doc'
        )


    @crew
    def crew(self) -> Crew:
        """Creates the MyCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
