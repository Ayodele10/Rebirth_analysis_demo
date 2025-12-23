import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# A random dataset simulating Rebirth 3.0 student data
np.random.seed(42)
n_students = 1000

data = {
    'student_id': range(1, n_students + 1),
    'login_frequency': np.random.randint(1, 30, n_students), # Logins in last month
    'task_completion_rate': np.random.uniform(0, 100, n_students), # % of tasks done
    'community_messages': np.random.randint(0, 150, n_students), # Engagement
    'fees_paid': np.random.choice([0, 1], n_students, p=[0.2, 0.8]) # 1 = Paid, 0 = Pending
}

df = pd.DataFrame(data)

# STUDENT SEGMENTATION USING K-MEANS

X = df[['login_frequency', 'task_completion_rate']]
kmeans = KMeans(n_clusters=3, n_init=10)
df['segment'] = kmeans.fit_predict(X)



def label_segment(row):
    if row['task_completion_rate'] > 70 and row['login_frequency'] > 20:
        return "Top Tier (Future Leaders)"
    elif row['task_completion_rate'] < 30:
        return "At-Risk (Need Nudge)"
    else:
        return "Steady Learners"

df['status'] = df.apply(label_segment, axis=1)

# THE RESULTS
print("--- REBIRTH 3.0 STUDENT SEGMENTATION ---")
print(df['status'].value_counts())

# Save to CSV for your GitHub
df.to_csv("rebirth_3_analytics.csv", index=False)
print("\nSuccess! 'rebirth_3_analytics.csv' created.")

# 4. QUICK VISUALIZATION
plt.figure(figsize=(10, 6))
for status in df['status'].unique():
    subset = df[df['status'] == status]
    plt.scatter(subset['login_frequency'], subset['task_completion_rate'], label=status)

plt.title("Rebirth 3.0: Intelligence Dashboard (Prototype)")
plt.xlabel("Login Frequency (Monthly)")
plt.ylabel("Task Completion (%)")
plt.legend()
plt.show()