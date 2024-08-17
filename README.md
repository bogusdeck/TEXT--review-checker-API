
<p align="center">
  <img src="https://github.com/user-attachments/assets/2ee9d05d-ee43-4ad9-baa0-b3393acd3e61" />
</p>

<p align="center">
https://reviewreward-production.up.railway.app
</p>


<div align="center">
<p>If you like my work, consider buying me a coffee! ☕️</p>
</div>


<div align="center">
<a href="https://www.buymeacoffee.com/bogusdeck" target="_blank">
    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee" width="150" />
</a>
</div>

# Review Authenticity Checker API

This project is a Flask-based API designed to check the authenticity of text reviews. It uses machine learning models to determine if a review is genuine or fake. The API supports both manual text review and automated checks using Cohere's language model.

## Features

- **Text Review Authenticity Check:** Determines if a given text review is real, fake, or computer-generated.
- **Machine Learning Models:** Utilizes Naive Bayes and SVM models trained on a dataset of deceptive and truthful reviews.
- **Cohere Language Model Integration:** Checks review authenticity using Cohere's language model for an additional layer of verification.

## Getting Started

### Prerequisites

- Python 3.11.5
- Flask
- Vercel CLI (for deployment)
- AWS S3 (optional, for remote model storage)

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/review-authenticity-checker.git
   cd review-authenticity-checker
   ```

2. Create a virtual environment:

   ```sh
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```sh
   pip install -r requirements.txt
   ```

4. Download NLTK stopwords:

   ```sh
   python -m nltk.downloader stopwords
   ```

### Configuration

1. **Cohere API Key:** Set up your Cohere API key as an environment variable to keep it secure:

   ```sh
   export COHERE_API_KEY="your-cohere-api-key"
   ```

2. **AWS S3 Configuration (Optional):** If you store your model files in AWS S3, configure your AWS credentials.

### Usage

1. **Run the Flask app:**

   ```sh
   flask run
   ```

2. **API Endpoint:**

   The API has one main endpoint:

   - `/api/review` (POST): Check the authenticity of a text review.

     **Request:**
     ```json
     {
       "productName": "Product Name",
       "category": "Category",
       "brand": "Brand",
       "purchaseDate": "2023-08-01",
       "purchasePrice": "100",
       "productReview": "This is my review text.",
       "shoppingLink": "http://shoppinglink.com"
     }
     ```

     **Response:**
     ```json
     {
       "message": "Original"  // or "Fake"
     }
     ```

### Deployment

1. **Create `.vercelignore` file:**

   ```plaintext
   .venv
   ```

2. **Generate `requirements.txt`:**

   ```sh
   pip freeze > requirements.txt
   ```

3. **Deploy to Vercel:**

   ```sh
   vercel
   ```

   Follow the prompts to complete the deployment process.

### Project Structure

```plaintext
your-project/
├── .vercelignore
├── requirements.txt
├── app.py
├── templates/
├── static/
└── ...
```

### Acknowledgements

- [Cohere](https://cohere.ai/) for their language model API.
- [NLTK](https://www.nltk.org/) for natural language processing tools.

```
