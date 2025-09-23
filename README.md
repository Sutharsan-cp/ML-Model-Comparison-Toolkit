# ML-Model-Comparison-Toolkit

## Overview

**ML-Model-Comparison-Toolkit** is a comprehensive toolkit for comparing, evaluating, and reporting on various machine learning models. It provides a backend for model training, evaluation, and reporting, and a frontend for user interaction and visualization.

---

## Features

- Upload and preprocess datasets
- Train multiple types of ML models (supervised, unsupervised, etc.)
- Compare model performance using standard metrics
- Generate and download detailed reports
- User-friendly web interface

---

## Project Structure

```
ML-Model-Comparison-Toolkit/
│
├── backend/
│   ├── app/
│   │   ├── api/                      # API endpoints: compare.py, recommend.py, report.py, train.py, upload.py
│   │   ├── api/preprocessing/        # Preprocessing modules (cleaning, scaling, encoding, pipelines)
│   │   ├── data/
│   │   │   ├── raw/                  # Original raw datasets (committed selectively)
│   │   │   ├── interim/              # Intermediate artifacts
│   │   │   └── processed/            # Cleaned/processed datasets used for training
│   │   ├── models/                   # Saved model artifacts and pipelines
│   │   │   ├── supervised/
│   │   │   ├── unsupervised/
│   │   │   ├── semi_supervised/
│   │   │   └── reinforcement/
│   │   ├── reports/                  # Generated reports & comparisons
│   │   ├── services/                 # Evaluator, trainer, recommender, report generator
│   │   └── utils/                    # Utilities: config_loader, eda_utils, statistics_utils, file_handler, logger
│   ├── config.yaml                   # Backend configuration
│   ├── requirements.txt              # Python dependencies
│   └── .env                          # Environment variables (not committed)
│
├── frontend/
│   ├── public/                       # Static files and index.html
│   ├── src/
│   │   ├── assets/                   # Images, icons, fonts
│   │   ├── components/               # Reusable React components (ChartCard, Loader, Navbar, Sidebar)
│   │   ├── context/                  # App context providers
│   │   ├── pages/                    # Views: Compare, Model, Preprocess, Recommend, Report, Upload, Profile
│   │   ├── services/                 # API client for backend
│   │   ├── styles/                   # Theme and CSS files
│   │   ├── utils/                    # Frontend helper utilities
│   │   ├── App.js                    # Main app component
│   │   └── index.js                  # React entry point
│   ├── package.json                  # Node dependencies and scripts
│   └── .env                          # Frontend env vars (not committed)
│
├── LICENSE
└── README.md
```

---

## Getting Started

### Backend

1. **Install dependencies**

   ```sh
   cd backend
   pip install -r requirements.txt
   ```

2. **Configure environment**

   - Edit `.env` and `config.yaml` as needed.

3. **Run the backend server**
   ```sh
   python app/main.py
   ```

### Frontend

1. **Install dependencies**

   ```sh
   cd frontend
   npm install
   ```

2. **Start the frontend**
   ```sh
   npm start
   ```

---

## Usage

1. Open the web interface in your browser.
2. Upload your dataset.
3. Select preprocessing options and models to train.
4. Compare results and download reports.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
