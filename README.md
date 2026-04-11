# bias_agent
SEN4018- Data Science with Python
Project Proposal

Team Members:
Eylül Çelen, 2202373
Mahir Gündoğan, 2284509

Motivation: The increasing use of AI systems in real-world decision-making such as hiring, healthcare, and finance has made data quality and fairness critically important. Many machine learning models unintentionally learn and amplify biases present in datasets, leading to unfair or discriminatory outcomes.
Our motivation for this project is to build an agentic AI system that can autonomously detect, analyze, and explain biases in datasets before they are used in model training. Instead of relying on manual inspection, this agent will actively investigate the dataset, decide what to analyze next, and provide interpretable results.

Kind of Problem: This project addresses a data-centric AI problem specifically, bias detection in structured datasets and automated exploratory data analysis. More specifically, it is a decision-making under uncertainty problem. The agent does not know in advance which columns are biased or how severe the imbalance is. It must use tools to gather evidence, reason about severity, prioritize its findings, and report its conclusions all without human guidance during the run.

What Problem Being Solved: Many datasets contain hidden biases such as, gender imbalance, age or socioeconomic skew or geographic underrepresentation. These biases can lead to discriminatory ML models, poor generalization or ethical and legal risks.
