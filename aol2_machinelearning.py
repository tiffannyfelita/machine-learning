import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class TurnoverClassifier:
    def __init__(self):
        self.model = None
        self.feature_names = [
            'Age', 'Gender', 'MaritalStatus', 'Travelling', 'Vertical', 'Qualification', 'EducationField', 'EmployeSatisfaction',
            'JobEngagement', 'JobLevel', 'JobSatisfaction', 'Role', 'DailyBilling', 'HourBilling', 'MonthlyBilling', 'MonthlyRate',
            'Work Experience', 'OverTime', 'PercentSalaryHike', 'Last Rating', 'RelationshipSatisfaction', 'Hours', 'StockOptionLevel',
            'TrainingTimesLastYear', 'Work&Life', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrentManager',
            'DistanceFromHome',
        ]
        self.df = None
        self.load_data()
        self.load_model()

    def load_data(self):
        try:
            self.df = pd.read_csv('turnover_data.csv')  # Pastikan kamu punya file ini
        except FileNotFoundError:
            self.df = None

    def train_model(self):
        if self.df is None:
            st.error("Dataset tidak ditemukan. Pastikan file 'turnover_data.csv' tersedia.")
            return

        X = self.df[self.feature_names]
        y = self.df['Turnover']  # Label target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        st.subheader("Model Performance")
        st.write(f"**Accuracy:** {acc:.2f}")
        st.text("Classification Report:")
        st.text(report)

        with open("rf_model.pkl", "wb") as f:
            pickle.dump(model, f)
        st.success("✅ Model berhasil dilatih dan disimpan.")

    def load_model(self):
        try:
            with open("rf_model.pkl", "rb") as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            self.model = None

    def predict(self, input_data):
        if self.model is None:
            return "Model belum dilatih."
        input_array = np.array(input_data).reshape(1, -1)
        prediction = self.model.predict(input_array)
        return "Yes" if prediction[0] == 1 else "No"


def main():
    st.title("💼 Employee Turnover Prediction App")
    st.sidebar.title("Menu")
    menu = ["Home", "Train Model", "Make Prediction"]
    choice = st.sidebar.selectbox("Navigation", menu)

    classifier = TurnoverClassifier()

    if choice == "Home":
        st.subheader("Welcome 👋")
        st.write("Gunakan aplikasi ini untuk melatih model dan memprediksi turnover karyawan.")

    elif choice == "Train Model":
        st.subheader("Train the Model")
        classifier.train_model()

    elif choice == "Make Prediction":
        st.subheader("Predict Employee Turnover")

        age = st.slider("Age", 18, 60, 30)
        gender = st.selectbox("Gender", [0, 1])
        marital_status = st.selectbox("Marital Status", [0, 1, 2])
        travelling = st.selectbox("Travelling", [0, 1, 2])
        vertical = st.selectbox("Vertical", [0, 1, 2])
        qualification = st.selectbox("Qualification", [0, 1, 2, 3])
        education_field = st.selectbox("Education Field", [0, 1, 2, 3])
        satisfaction = st.slider("Employee Satisfaction", 1, 5, 3)
        engagement = st.slider("Job Engagement", 1, 5, 3)
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        job_satisfaction = st.slider("Job Satisfaction", 1, 5, 3)
        role = st.selectbox("Job Role", [0, 1, 2, 3])
        daily_billing = st.number_input("Daily Billing", 100, 1000, 300)
        hour_billing = st.slider("Hourly Billing", 10, 100, 40)
        monthly_billing = st.number_input("Monthly Billing", 1000, 20000, 5000)
        monthly_rate = st.number_input("Monthly Rate", 1000, 20000, 8000)
        work_exp = st.slider("Work Experience (years)", 0, 40, 5)
        overtime = st.selectbox("OverTime", [0, 1])
        salary_hike = st.slider("Percent Salary Hike", 0, 30, 10)
        last_rating = st.slider("Last Performance Rating", 1, 5, 3)
        rel_satisfaction = st.slider("Relationship Satisfaction", 1, 5, 3)
        hours = st.slider("Working Hours", 1, 24, 8)
        stock_option = st.selectbox("Stock Option Level", [0, 1])
        training = st.slider("Training Times Last Year", 0, 10, 3)
        work_life = st.slider("Work-Life Balance", 1, 4, 3)
        years_at_company = st.slider("Years At Company", 0, 40, 5)
        in_role = st.slider("Years In Current Role", 0, 20, 2)
        since_promo = st.slider("Years Since Last Promotion", 0, 15, 1)
        with_manager = st.slider("Years With Current Manager", 0, 20, 2)
        distance = st.slider("Distance From Home", 1, 30, 5)

        input_data = [
            age, gender, marital_status, travelling, vertical, qualification, education_field, satisfaction,
            engagement, job_level, job_satisfaction, role, daily_billing, hour_billing, monthly_billing, monthly_rate,
            work_exp, overtime, salary_hike, last_rating, rel_satisfaction, hours, stock_option,
            training, work_life, years_at_company, in_role, since_promo, with_manager, distance
        ]

        if st.button("Predict"):
            result = classifier.predict(input_data)
            st.success(f"Prediction: The employee is likely to leave? **{result}**")


if __name__ == "__main__":
    main()

