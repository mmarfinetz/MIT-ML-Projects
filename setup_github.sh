#!/bin/bash

# Setup script for GitHub repository
echo "🚀 MIT ML Projects - GitHub Setup Script"
echo "========================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}This script will help you push your MIT ML projects to GitHub${NC}"
echo ""

# Instructions for creating repository on GitHub
echo -e "${YELLOW}Step 1: Create a new repository on GitHub${NC}"
echo "1. Go to https://github.com/new"
echo "2. Name it: MIT-ML-Projects"
echo "3. Make it Public (to showcase your work)"
echo "4. Don't initialize with README (we already have one)"
echo "5. Click 'Create repository'"
echo ""

read -p "Press Enter when you've created the repository..."

# Get GitHub username
echo ""
read -p "Enter your GitHub username: " GITHUB_USERNAME

# Repository URL
REPO_URL="https://github.com/${GITHUB_USERNAME}/MIT-ML-Projects.git"

echo ""
echo -e "${BLUE}Setting up remote repository...${NC}"

# Add remote origin
git remote add origin $REPO_URL 2>/dev/null || git remote set-url origin $REPO_URL

# Initial commit
echo -e "${BLUE}Creating initial commit...${NC}"
git add -A
git commit -m "Initial commit: MIT Machine Learning Projects Portfolio

- Used Car Price Prediction (Regression)
- Boston Housing Prediction
- Classification Projects (ML & PDS)
- Time Series Analysis
- Unsupervised Learning
- SVHN Deep Learning (CNN & NN)

All projects completed as part of MIT Professional Education programs."

# Push to GitHub
echo ""
echo -e "${BLUE}Pushing to GitHub...${NC}"
git branch -M main
git push -u origin main

echo ""
echo -e "${GREEN}✅ Success! Your projects are now on GitHub!${NC}"
echo ""
echo "📊 Repository URL: $REPO_URL"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Visit your repository: $REPO_URL"
echo "2. Add project topics/tags (machine-learning, deep-learning, data-science, mit)"
echo "3. Consider adding:"
echo "   - Jupyter notebooks (convert HTML files)"
echo "   - Sample datasets (if not too large)"
echo "   - Model performance visualizations"
echo "   - Links to your LinkedIn/portfolio"
echo ""
echo -e "${BLUE}Optional: Create individual repositories${NC}"
echo "If you want each project as a separate repo, run:"
echo "  ./create_individual_repos.sh"
echo ""
echo -e "${GREEN}🎉 Congratulations on completing your MIT ML projects!${NC}"