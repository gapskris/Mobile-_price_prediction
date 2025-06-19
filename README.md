# Mobile-_price_prediction

Mobile Price Prediction
This project aims to predict the price range of mobile phones based on their technical specifications using supervised machine learning. The model classifies phones into four price segments, helping businesses and consumers estimate market positioning.

📊 Problem Statement
Mobile manufacturers and e-commerce platforms need to estimate the appropriate price category of mobile phones based on specifications. Automating this process helps in:

Product pricing decisions

Market analysis and segmentation

Enhancing recommendation systems

📁 Dataset
The dataset used for this project is from Kaggle - Mobile Price Classification. It includes mobile features like battery power, RAM, screen size, camera specs, and others.

Feature	Description
battery_power	Total energy capacity of the battery
blue	Bluetooth enabled or not (0 or 1)
clock_speed	Processor speed
dual_sim	Supports dual SIM (0 or 1)
fc	Front Camera megapixels
four_g	Supports 4G (0 or 1)
int_memory	Internal Memory in GB
m_dep	Mobile depth in cm
mobile_wt	Weight of mobile in grams
n_cores	Number of processor cores
pc	Primary camera megapixels
px_height	Pixel Resolution Height
px_width	Pixel Resolution Width
ram	RAM in MB
sc_h	Screen Height
sc_w	Screen Width
talk_time	Battery life during calls
three_g	Supports 3G (0 or 1)
touch_screen	Touch screen available (0 or 1)
wifi	WiFi capability (0 or 1)
price_range (target)	0 = Low, 1 = Medium, 2 = High, 3 = Very High

⚙️ Technologies Used
Python 3.x

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

XGBoost, LightGBM (optional)

Jupyter Notebook / Streamlit

🧠 Model Workflow
Data Preprocessing

Check for missing values (none in this dataset)

Feature scaling (e.g., StandardScaler or MinMaxScaler)

Train-test split

Model Training

Logistic Regression

Random Forest

Gradient Boosting / XGBoost / LightGBM

Hyperparameter tuning with GridSearchCV / RandomizedSearchCV

Model Evaluation

Accuracy

Confusion Matrix

Classification Report (Precision, Recall, F1 Score)

Deployment (Optional)

Streamlit app to enter mobile specs and predict price range

🧪 How to Run
Clone the repo:

bash
Copy
Edit
git clone https://github.com/yourusername/mobile-price-prediction.git
cd mobile-price-prediction
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the notebook:

bash
Copy
Edit
jupyter notebook Mobile_Price_Prediction.ipynb
(Optional) Launch Streamlit app:

bash
Copy
Edit
streamlit run app.py
📊 Results
Best Model: Random Forest Classifier

Accuracy: ~92%

Precision/Recall: High for all 4 price categories

🔍 Insights
RAM, battery power, and pixel resolution are the top predictors

Features like dual SIM and WiFi have less influence

📌 Future Work
Add brand or OS features

Build price regression model (predict exact price instead of category)

Integrate with mobile product APIs for real-time input

Add SHAP for explainable AI

