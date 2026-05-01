# bias_agent
SEN4018- Data Science with Python
Project Proposal

Team Members:
Eylül Çelen, 2202373
Mahir Gündoğan, 2284509

Motivation: The increasing use of AI systems in real-world decision-making such as hiring, healthcare, and finance has made data quality and fairness critically important. Many machine learning models unintentionally learn and amplify biases present in datasets, leading to unfair or discriminatory outcomes. Our motivation for this project is to build an agentic AI system that can autonomously detect, analyze, and explain biases in datasets before they are used in model training. Instead of relying on manual inspection, this agent will actively investigate the dataset, decide what to analyze next, and provide interpretable results.

Kind of Problem: This project addresses a data-centric AI problem specifically, bias detection in structured datasets and automated exploratory data analysis. More specifically, it is a decision-making under uncertainty problem. The agent does not know in advance which columns are biased or how severe the imbalance is. It must use tools to gather evidence, reason about severity, prioritize its findings, and report its conclusions all without human guidance during the run.

What Problem Being Solved: Many datasets contain hidden biases such as, gender imbalance, age or socioeconomic skew or geographic underrepresentation. These biases can lead to discriminatory ML models, poor generalization or ethical and legal risks.

Technical Approach: This project will be implemented as an end-to-end agentic system using Python, where the user interacts with the system through a simple web interface built with Gradio. The backend will rely on pandas for data processing and libraries such as matplotlib and seaborn for visualization. Once a CSV file is uploaded, the agent will automatically inspect the dataset, determine column types, and decide which analyses to perform. The system will include an autonomous decision loop where the agent iteratively selects actions such as distribution analysis, cross-feature comparison, or visualization based on intermediate findings. Additionally, a language model integrated via Hugging Face will be used to generate human-readable explanations and to evaluate whether the detected patterns are meaningful or potentially misleading.

Datasets: The system is designed to operate on structured CSV dataset provided by the user however, to test and validate the effectiveness of the agent, several benchmark datasets will be used during development. These include the Titanic dataset, which demonstrates real-world distribution differences, as well as datasets like Adult Income and synthetically generated biased datasets such other datasets will be derived from Kaggle as CSV file.

Team Responsibilities: Team member Eylül (2202373) will focus on the core system implementation, including data processing, bias detection algorithms, and the agent decision loop. The other member Mahir (2284509) will handle the user interface, visualization components, and integration of the language model for explanation and evaluation. Both team members will jointly contribute to testing the system on multiple datasets, debugging, preparing the documentation, and deploying the final application on Hugging Face Spaces.
