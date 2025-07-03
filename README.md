# Ultimate Load Prediction in A36 Steel using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Project Description

This project develops a Machine Learning model to predict the ultimate load in A36 steel tensile tests, using the physical dimensions of the samples as input variables. A36 steel is one of the most widely used materials in the construction and structural engineering industry due to its excellent mechanical properties and cost-effectiveness.

The ability to accurately predict the ultimate load that a steel sample can withstand before fracture is fundamental for:
- Optimizing the design of structural components
- Reducing the need for extensive destructive testing
- Improving quality and safety in manufacturing processes
- Accelerating product development in the metallurgical industry

## 🎯 Objectives

### Main Objective
Develop a robust and accurate predictive model that estimates the ultimate tensile load in A36 steel samples based on their physical dimensions.

### Specific Objectives
- Perform comprehensive exploratory analysis of the tensile test dataset
- Implement feature engineering techniques to improve model performance
- Evaluate multiple Machine Learning algorithms to select the most suitable one
- Validate the model using standard regression metrics
- Create professional visualizations that facilitate result interpretation

## 📊 Dataset

The dataset contains **20,000 records** of tensile tests performed on A36 steel samples, with the following characteristics:

| Variable | Description | Unit | Type |
|----------|-------------|------|------|
| `Length (cm)` | Longitudinal dimension of the sample | cm | Numeric |
| `Width (cm)` | Transverse dimension of the sample | cm | Numeric |
| `Thickness (cm)` | Sample thickness | cm | Numeric |
| `Ultimate Load (kgf)` | Maximum load supported before fracture | kgf | Numeric (Target) |

### Derived Features
During the feature engineering process, the following additional variables were generated:

- **Area (cm²)**: Calculated as `Width × Thickness`
- **Stress (Kg/cm²)**: Calculated as `Ultimate Load / Area`

These derived features provide valuable information about the material's mechanical properties and significantly improve the model's predictive capacity.

## 🔧 Technologies Used

### Languages and Frameworks
- **Python 3.8+**: Main development language
- **Jupyter Notebook**: Interactive development environment

### Machine Learning Libraries
- **Scikit-learn**: ML algorithm implementation and evaluation metrics
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations and linear algebra

### Data Visualization
- **Matplotlib**: Static graph creation
- **Seaborn**: Advanced statistical visualizations

### Development Tools
- **Git**: Version control
- **GitHub**: Remote repository and collaboration

## 🚀 Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/MFelix9310/a36-steel-prediction.git
cd a36-steel-prediction
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('All dependencies installed successfully')"
```

## 📁 Project Structure

```
a36-steel-prediction/
│
├── data/
│   ├── raw/
│   │   └── A36_tensile_tests.csv
│   └── processed/
│       └── processed_data.csv
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── visualization.py
│
├── models/
│   ├── random_forest_model.pkl
│   └── model_metrics.json
│
├── visualizations/
│   ├── stress_distribution.png
│   ├── ultimate_load_vs_area.png
│   ├── stress_boxplot.png
│   └── variables_pairplot.png
│
├── docs/
│   ├── A36_steel_presentation.pdf
│   └── technical_report.md
│
├── requirements.txt
├── README.md
└── LICENSE
```

## 🔍 Exploratory Data Analysis

### Descriptive Statistics

The initial analysis revealed the following dataset characteristics:

- **Dataset size**: 20,000 samples
- **Numeric variables**: 4 (3 input + 1 target)
- **Missing values**: 0 (complete dataset)
- **Target variable range**: 4,078.86 - 5,608.44 kgf

### Variable Distribution

The stress distribution (Kg/cm²) shows a near-normal trend, with:
- **Mean**: 4,845.23 Kg/cm²
- **Standard deviation**: 253.52 Kg/cm²
- **Median**: 4,843.67 Kg/cm²

This distribution indicates that most samples present stress values concentrated around the mean, which is expected for a homogeneous material like A36 steel.

### Correlations

Correlation analysis revealed:
- **Strong correlation** between Area and Ultimate Load (r > 0.9)
- **Moderate correlation** between individual dimensions and target variable
- **Low correlation** between input variables, indicating little multicollinearity

## 🤖 Machine Learning Methodology

### Algorithm Selection

After evaluating multiple algorithms, **Random Forest Regressor** was selected for the following reasons:

1. **Robustness**: Less prone to overfitting compared to individual decision trees
2. **Handling non-linear relationships**: Captures complex patterns in data
3. **Interpretability**: Provides information about feature importance
4. **Performance**: Excellent balance between accuracy and generalization

### Model Configuration

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
```

### Data Split

- **Training**: 80% (16,000 samples)
- **Testing**: 20% (4,000 samples)
- **Cross-validation**: 5-fold for hyperparameter optimization

## 📈 Results and Evaluation

### Performance Metrics

The Random Forest Regressor model demonstrated excellent performance:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.9847 | The model explains 98.47% of the variance |
| **MSE** | 2,847.32 | Low mean squared error |
| **RMSE** | 53.35 kgf | Average error of ±53.35 kgf |
| **MAE** | 41.22 kgf | Mean absolute error of 41.22 kgf |

### Feature Importance

The importance analysis revealed that the most relevant features are:

1. **Area (cm²)**: 45.2% - Most determining factor
2. **Thickness (cm)**: 28.7% - Second most important feature
3. **Width (cm)**: 15.8% - Moderate contribution
4. **Length (cm)**: 10.3% - Relatively minor influence

### Model Validation

Cross-validation confirmed model stability:
- **Average R²**: 0.9841 ± 0.0012
- **Average RMSE**: 54.12 ± 2.18 kgf

These results indicate that the model is robust and generalizes well to unseen data.

## 📊 Visualizations

The project includes professional visualizations that facilitate result interpretation:

### 1. Stress Distribution
![Stress Distribution](visualizations/stress_distribution.png)

### 2. Ultimate Load vs. Area Relationship
![Ultimate Load vs. Area](visualizations/ultimate_load_vs_area.png)

### 3. Scatter Analysis (Boxplot)
![Stress Boxplot](visualizations/stress_boxplot.png)

### 4. Correlation Matrix (Pairplot)
![Variables Pairplot](visualizations/variables_pairplot.png)

## 🚀 Model Usage

### Individual Prediction

```python
import pickle
import numpy as np

# Load the trained model
with open('models/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example prediction
length = 20.5  # cm
width = 7.2    # cm
thickness = 1.8  # cm

# Prepare input data
X_new = np.array([[length, width, thickness]])

# Make prediction
predicted_load = model.predict(X_new)[0]
print(f"Predicted ultimate load: {predicted_load:.2f} kgf")
```

### Batch Prediction

```python
import pandas as pd

# Load new data
new_data = pd.read_csv('new_samples.csv')

# Make predictions
predictions = model.predict(new_data[['Length (cm)', 'Width (cm)', 'Thickness (cm)']])

# Add predictions to DataFrame
new_data['Predicted_Load'] = predictions
```

## 🔮 Future Work

### Proposed Improvements

1. **Advanced Algorithm Exploration**
   - Implementation of deep neural networks
   - Evaluation of gradient boosting algorithms (XGBoost, LightGBM)
   - Comparison with more complex ensemble models

2. **Advanced Feature Engineering**
   - Incorporation of steel chemical properties
   - Analysis of polynomial features and interactions
   - Implementation of automatic feature selection techniques

3. **External Validation**
   - Testing with datasets from other steel types
   - Validation with independent experimental data
   - Model transferability analysis

4. **Application Development**
   - Creation of an interactive web interface
   - REST API for integration with existing systems
   - Mobile application for field use

### Project Extensions

- **Uncertainty Analysis**: Implementation of confidence intervals in predictions
- **Anomaly Detection**: Automatic identification of samples with atypical behavior
- **Multi-objective Optimization**: Balance between accuracy and computational time

## 🤝 Contributions

Contributions are welcome and appreciated. To contribute:

1. **Fork** the repository
2. Create a **branch** for your feature (`git checkout -b feature/new-feature`)
3. **Commit** your changes (`git commit -am 'Add new feature'`)
4. **Push** to the branch (`git push origin feature/new-feature`)
5. Open a **Pull Request**

### Contribution Guidelines

- Follow PEP 8 coding conventions
- Include tests for new functionalities
- Update documentation as necessary
- Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## 👨‍💻 Author

**Félix Ruiz**  
*Data Scientist & Machine Learning Specialist*

- Portfolio: [gambito93.pythonanywhere.com](https://gambito93.pythonanywhere.com/)
- LinkedIn: [felix-ruiz-muñoz-data-science](https://www.linkedin.com/in/felix-ruiz-mu%C3%B1oz-data-science/)
- GitHub: [MFelix9310](https://github.com/MFelix9310)

## 🙏 Acknowledgments

- To the data science community for the tools and libraries used
- To materials science researchers for providing theoretical context
- To all contributors who have helped improve this project

## 📚 References

1. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
2. [Random Forest Algorithm](https://en.wikipedia.org/wiki/Random_forest)
3. [ASTM Standards for Tensile Testing](https://www.astm.org/)
4. [Steel Material Properties](https://www.steelconstruction.info/)

---

⭐ If this project has been useful to you, don't forget to give it a star on GitHub!
