#!/usr/bin/env python3
"""
Convert HTML Jupyter notebooks back to .ipynb format and organize them
"""

import os
import shutil
from pathlib import Path

# Mapping of HTML files to their respective projects
notebook_mapping = {
    "01-Regression-Used-Car-Pricing": {
        "files": [],  # Will use the PDFs for this project
        "notebook_name": "used_car_price_prediction.ipynb",
        "description": "Used Car Price Prediction using Regression Models"
    },
    "02-Boston-Housing-Prediction": {
        "files": ["Learners_Notebook_Boston_house_price (1).html"],
        "notebook_name": "boston_housing_prediction.ipynb",
        "description": "Boston Housing Price Prediction"
    },
    "03-Classification-ML": {
        "files": ["Learner+Notebook+-+Project_Classification_ML.html"],
        "notebook_name": "classification_ml.ipynb",
        "description": "Machine Learning Classification Project"
    },
    "04-Classification-PDS": {
        "files": ["Learner_Notebook_Classification_PDS.html"],
        "notebook_name": "classification_pds.ipynb",
        "description": "Classification - Practical Data Science"
    },
    "05-Time-Series-Analysis": {
        "files": ["Learner_Notebook_Project_Time_Series.html"],
        "notebook_name": "time_series_analysis.ipynb",
        "description": "Time Series Analysis and Forecasting"
    },
    "06-Unsupervised-Learning": {
        "files": ["Learner_Notebook_Unsupervised_Learning_Project_(1) (1).html"],
        "notebook_name": "unsupervised_learning.ipynb",
        "description": "Unsupervised Learning - Clustering and Dimensionality Reduction"
    },
    "07-SVHN-Deep-Learning": {
        "files": ["CNN_Project_Learner_Notebook_SVHN (1).html", 
                  "NN_Project_Learner_Notebook_SVHN (2).html"],
        "notebook_name": "svhn_deep_learning.ipynb",
        "description": "Street View House Numbers Classification using Deep Learning"
    }
}

def copy_files():
    """Copy HTML files to their respective project folders"""
    source_dir = Path("/Users/mitch/Desktop/MIT Projects")
    target_dir = Path("/Users/mitch/Desktop/MIT-ML-Projects")
    
    for project, info in notebook_mapping.items():
        project_dir = target_dir / project
        
        # Copy HTML files
        for html_file in info["files"]:
            source_file = source_dir / html_file
            if source_file.exists():
                target_file = project_dir / html_file
                shutil.copy2(source_file, target_file)
                print(f"Copied {html_file} to {project}")
            else:
                print(f"Warning: {html_file} not found")
    
    # Copy PDFs
    pdf_files = [
        ("Final Project (2).pdf", "01-Regression-Used-Car-Pricing", "regression_analysis_report.pdf"),
        ("Untitled document (1) (1).pdf", "01-Regression-Used-Car-Pricing", "milestone1_report.pdf")
    ]
    
    for pdf, project, new_name in pdf_files:
        source_file = source_dir / pdf
        if source_file.exists():
            target_file = target_dir / project / new_name
            shutil.copy2(source_file, target_file)
            print(f"Copied {pdf} to {project} as {new_name}")

def create_requirements_files():
    """Create requirements.txt for each project"""
    target_dir = Path("/Users/mitch/Desktop/MIT-ML-Projects")
    
    base_requirements = """# Base requirements for machine learning projects
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
jupyter>=1.0.0
ipython>=7.0.0
"""

    ml_requirements = base_requirements + """# Additional ML requirements
xgboost>=1.4.0
scipy>=1.7.0
statsmodels>=0.12.0
"""

    deep_learning_requirements = ml_requirements + """# Deep Learning requirements
tensorflow>=2.6.0
keras>=2.6.0
opencv-python>=4.5.0
pillow>=8.3.0
"""
    
    # Define which projects need which requirements
    project_requirements = {
        "01-Regression-Used-Car-Pricing": ml_requirements,
        "02-Boston-Housing-Prediction": ml_requirements,
        "03-Classification-ML": ml_requirements,
        "04-Classification-PDS": ml_requirements,
        "05-Time-Series-Analysis": ml_requirements + "prophet>=1.0.0\nplotly>=5.0.0\n",
        "06-Unsupervised-Learning": ml_requirements,
        "07-SVHN-Deep-Learning": deep_learning_requirements
    }
    
    for project, reqs in project_requirements.items():
        req_file = target_dir / project / "requirements.txt"
        with open(req_file, 'w') as f:
            f.write(reqs)
        print(f"Created requirements.txt for {project}")

def create_gitignore():
    """Create .gitignore file for the main repository"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# Data files (usually too large for git)
*.csv
*.xlsx
*.xls
data/
datasets/

# Model files
*.pkl
*.h5
*.model
*.joblib
models/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Temporary files
*.tmp
temp/
tmp/

# Documentation build
docs/_build/
"""
    
    target_file = Path("/Users/mitch/Desktop/MIT-ML-Projects/.gitignore")
    with open(target_file, 'w') as f:
        f.write(gitignore_content)
    print("Created .gitignore file")

def main():
    print("Organizing MIT ML Projects for GitHub...")
    print("-" * 50)
    
    # Copy files to organized structure
    print("\n1. Copying files to project folders...")
    copy_files()
    
    # Create requirements files
    print("\n2. Creating requirements.txt files...")
    create_requirements_files()
    
    # Create gitignore
    print("\n3. Creating .gitignore...")
    create_gitignore()
    
    print("\n✅ Project organization complete!")
    print("\nNext steps:")
    print("1. Convert HTML files to .ipynb using Jupyter's nbconvert or manual extraction")
    print("2. Review and enhance notebook content with analysis from PDFs")
    print("3. Initialize git repositories")
    print("4. Push to GitHub")

if __name__ == "__main__":
    main()