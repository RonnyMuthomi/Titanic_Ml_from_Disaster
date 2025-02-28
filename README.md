# Titanic_Ml_from_Disaster
Prediction of Titanic survivors
# Titanic Survival Prediction - Machine Learning Model

## 📌 Project Overview
This project aims to build a **predictive model** for determining whether a passenger survived or not based on the famous **Titanic dataset**. The dataset contains various passenger attributes such as age, sex, ticket class, and more. The goal is to apply **machine learning techniques** to develop an accurate classification model.

## 📂 Dataset Information
The dataset consists of the following key features:

- **Survived** (Target Variable: `0` = No, `1` = Yes)
- **Pclass** (Ticket class: `1st`, `2nd`, `3rd`)
- **Sex** (Gender: Male, Female)
- **Age** (Age of passengers)
- **SibSp** (# of siblings/spouses aboard)
- **Parch** (# of parents/children aboard)
- **Fare** (Ticket price)
- **Embarked** (Port of embarkation: C, Q, S)

The dataset is sourced from [Kaggle's Titanic Dataset](https://www.kaggle.com/c/titanic/data).

## 🚀 Project Workflow
1. **Problem Understanding**: Define objectives and analyze data requirements.
2. **Data Collection & Loading**: Load Titanic dataset using `pandas`.
3. **Exploratory Data Analysis (EDA)**:
   - Summary statistics (`describe()`, `info()`)
   - Missing values analysis and imputation
   - Visualizations (`seaborn`, `matplotlib`)
4. **Feature Engineering**:
   - Handling missing values (e.g., imputing `Age`, `Embarked`)
   - Encoding categorical features (`OneHotEncoder`)
   - Scaling numerical features (`StandardScaler`)
5. **Model Selection & Training**:
   - Train models (e.g., `Logistic Regression`, `Random Forest`, `XGBoost`)
   - Hyperparameter tuning (`GridSearchCV`)
6. **Model Evaluation**:
   - Metrics: `accuracy_score`, `precision`, `recall`, `f1-score`
   - ROC Curve and Confusion Matrix analysis
7. **Model Deployment**:
   - Export best model using `joblib`
   - Deploy using `Flask` or `Streamlit`

## ⚙️ Technologies & Libraries Used
- **Programming Language**: Python (Jupyter Notebook / Google Colab)
- **Libraries**:
  - `pandas`, `numpy` (Data Processing)
  - `seaborn`, `matplotlib` (Data Visualization)
  - `scikit-learn` (Machine Learning)
  - `XGBoost` (Advanced ML Model)
  - `Flask` / `Streamlit` (Deployment)

## 📊 Model Performance
| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|------------|--------|----------|
| Logistic Regression | 80.2%   | 78.5%       | 70.3%  | 74.1%    |
| Random Forest       | 84.6%   | 81.3%       | 75.2%  | 78.1%    |
| XGBoost            | **86.1%**  | **83.7%**   | **78.6%**  | **81.0%**    |

## 📌 How to Run the Project
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/titanic-ml.git
   cd titanic-ml
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook for model training.
4. To deploy, use Flask or Streamlit:
   ```sh
   python app.py  # Flask
   streamlit run app.py  # Streamlit
   ```

## 📌 Future Improvements
- Improve feature engineering (e.g., advanced feature selection, feature interactions).
- Test additional ML models (e.g., `CatBoost`, `Neural Networks`).
- Optimize hyperparameters further for better accuracy.
- Deploy as a web-based ML service.

## 👨‍💻 Author
**Your Name**  
📧 Email: ronnymuthomi254@gmail.com
🔗 LinkedIn: www.linkedin.com/in/ronny-muthomi-014805262 
🔗 GitHub: 

---
### ⭐ If you find this project helpful, please give it a star ⭐

