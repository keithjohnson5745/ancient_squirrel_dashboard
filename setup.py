from setuptools import setup, find_packages

setup(
    name="ancient_squirrel",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "nltk",
        "gensim",
        "tqdm",
        "spacy",
        "streamlit",
        "plotly",
        "matplotlib",
        "wordcloud",
        "pyyaml"
    ],
    extras_require={
        "dashboard": ["streamlit>=1.20.0", "wordcloud", "matplotlib", "plotly"],
        "nlp": ["spacy>=3.0.0", "nltk>=3.6.0", "gensim>=4.0.0"],
        "openai": ["openai>=1.0.0"],
        "full": [
            "streamlit>=1.20.0", "wordcloud", "matplotlib", "plotly",
            "spacy>=3.0.0", "nltk>=3.6.0", "gensim>=4.0.0",
            "openai>=1.0.0", "sentence-transformers"
        ]
    },
    entry_points={
        "console_scripts": [
            "ancient-analyze=ancient_squirrel.cli.analyze:main",
            "ancient-dashboard=ancient_squirrel.cli.dashboard:main",
        ],
    },
    python_requires=">=3.8",
    description="YouTube Network Analysis and Dashboard",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/ancient_squirrel_dashboard",
)