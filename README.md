# Irrigation Pump Prediction

This project provides a **machine learning model to predict irrigation pump status** based on environmental factors like moisture and temperature. It includes two models:
- **Random Forest Classifier** (`irrigation_rf_classifier.py`)
- **Gradient Boosting Classifier** (`irrigation_gb_classifier.py`)

Both models are built using **Streamlit** for an interactive UI.

## 📌 Features
✔️ Load and visualize irrigation data  
✔️ Train models with Random Forest or Gradient Boosting  
✔️ Predict whether the irrigation pump should be ON or OFF  
✔️ Upload custom datasets for predictions  
✔️ Tune hyperparameters for better model performance  
✔️ Download the predictions  

## 🛠 Installation
Ensure you have **Python 3.7+** installed. Then, install dependencies:

```bash
pip install -r requirements.txt
```

To start the Streamlit app, run:

```bash
streamlit run irrigation_rf_classifier.py
```
or  
```bash
streamlit run irrigation_gb_classifier.py
```

## 📂 Dataset Format
Your CSV should have the following columns:
- `moisture` (numeric)
- `temperature` (numeric)
- `crop` (categorical: e.g., Wheat, Cotton)
- `pump` (target: 0 = OFF, 1 = ON)

Example:
| moisture | temperature | crop   | pump |
|----------|------------|--------|------|
| 650      | 25         | Wheat  | 1    |
| 400      | 30         | Cotton | 0    |

## 📊 Model Comparison
| Model                 | Accuracy (Example) |
|----------------------|-----------------|
| Random Forest       | **92.3%**       |
| Gradient Boosting   | **94.1%**       |

## 🚀 Future Enhancements
- Add more crop types for better predictions
- Include weather forecast data
- Add real-time sensor integration

## 📜 License
This project is licensed under the MIT License.
