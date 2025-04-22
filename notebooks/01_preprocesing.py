from google.colab import drive
drive.mount('/content/drive')


import os

project_root = "/content/drive/MyDrive/FraudBehaviorEmbeddings"
subfolders = [
    "notebooks", "data/raw", "data/processed",
    "models/transformer", "models/graph", "models/autoencoder",
    "utils"
]

for folder in subfolders:
    os.makedirs(os.path.join(project_root, folder), exist_ok=True)

# print("Project structure created in Google Drive ✅")


# import pandas as pd
# import random
# from datetime import datetime, timedelta

# Simulated data
# users = [f"user_{i}" for i in range(5)]
# actions = ["open_app", "switch_app", "send_msg", "join_group", "logout"]

# data = []

# for user in users:
#     timestamp = datetime(2025, 4, 1)
#     for _ in range(10):  # 10 events per user
#         action = random.choice(actions)
#         data.append({
#             "user_id": user,
#             "action": action,
#             "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S')
#         })
#         timestamp += timedelta(minutes=random.randint(50, 59))

# df = pd.DataFrame(data)
# df.to_csv(f"{project_root}/data/raw/user_behavior.csv", index=False)
# df.head()


from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

cdr_path = os.path.join(project_root, "data/raw/Simulated_CDR_IPDR_Logs.csv")
rtm_path = os.path.join(project_root, "data/raw/Simulated_RTM_Logs.csv")

cdr_df = pd.read_csv(cdr_path)
rtm_df = pd.read_csv(rtm_path)

# Peek into the data
print(" CDR/IPDR Dataset Sample:")
print(cdr_df.head())

print("\n RTM App Usage Dataset Sample:")
print(rtm_df.head())

print("\n CDR Columns:", list(cdr_df.columns))
print(" RTM Columns:", list(rtm_df.columns))

print("\n Fraud Label Distribution (CDR):")
print(cdr_df['fraud_label'].value_counts())

print("\n Fraud Label Distribution (RTM):")
print(rtm_df['fraud_label'].value_counts())



# print(cdr_df.isnull().sum())#no null records
# print(rtm_df.isnull().sum())

# print(rtm_df['timestamp'])
# print(cdr_df['duration_sec']) #already in proper format
# to create a proper pipeline, let's do preprocessing
cdr_df['timestamp'] = pd.to_datetime(cdr_df['timestamp'])
cdr_df['duration_sec'] = cdr_df['duration_sec'].astype(int)#convert into integer

rtm_df['timestamp'] = pd.to_datetime(rtm_df['timestamp'])


# # 📌 Save processed versions :
cdr_df.to_csv(os.path.join(project_root, "data/processed/CDR_cleaned.csv"), index=False)
rtm_df.to_csv(os.path.join(project_root, "data/processed/RTM_cleaned.csv"), index=False)

#read precessed data
cdr_df = pd.read_csv(cdr_path)
rtm_df = pd.read_csv(rtm_path)
# print(cdr_df['timestamp'].dtypes)# this is a string object
#convert timestamps
# values in the timestamp column of the DataFrame
# cdr_df from string format (or any other format) into datetime format using pandas.to_datetime().
cdr_df['timestamp'] = pd.to_datetime(cdr_df['timestamp'])
rtm_df['timestamp'] = pd.to_datetime(rtm_df['timestamp'])
# print(cdr_df['timestamp'].dtypes)
#  STEP 4: Add action tokens
cdr_df['action'] = cdr_df['type'] + "_" + cdr_df['destination'].str.extract(r'(\w+)\.')[0].fillna("unknown")
rtm_df['action'] = "app_" + rtm_df['app'].str.lower()
# print(cdr_df['action'])
# print(rtm_df['action'])




#  STEP 5: Merge CDR and RTM
combined_df = pd.concat([cdr_df[['subscriber_id', 'timestamp', 'action', 'fraud_label']],
                         rtm_df[['subscriber_id', 'timestamp', 'action', 'fraud_label']]],
                         ignore_index=True)

combined_df = combined_df.sort_values(by=['subscriber_id', 'timestamp']).reset_index(drop=True)
# print(combined_df)

#  STEP 6: Create user-wise sequences
user_sequences = combined_df.groupby('subscriber_id')['action'].apply(list).reset_index(name='action_sequence')
user_labels = combined_df.groupby('subscriber_id')['fraud_label'].agg(lambda x: x.mode()[0]).reset_index(name='label')

merged = pd.merge(user_sequences, user_labels, on='subscriber_id')
# merged.head()
print(merged)



from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenize actions
# action_le = LabelEncoder()
# merged['action_sequence_tokens'] = merged['action_sequence'].apply(lambda x: action_le.fit_transform(x))
action_le = LabelEncoder()
action_le.fit([item for sublist in merged['action_sequence'] for item in sublist])  # fit on all actions
merged['action_sequence_tokens'] = merged['action_sequence'].apply(lambda x: action_le.transform(x))

# Pad sequences
max_len = max(merged['action_sequence_tokens'].apply(len))
merged['padded_sequence'] = merged['action_sequence_tokens'].apply(lambda x: pad_sequences([x], maxlen=max_len, padding='post')[0])

# Encode labels
label_map = {'Benign': 0, 'Psychological': 1, 'Technical': 2}  # Add more if needed
merged['label_encoded'] = merged['label'].map(label_map)
# print(merged)
#  STEP 8: Save for Transformer training
transformer_input_df = merged[['subscriber_id', 'padded_sequence', 'label_encoded']]
print(transformer_input_df)
transformer_input_df.to_pickle(os.path.join(project_root, "data/processed/transformer_input.pkl"))

print(" Saved transformer_input.pkl")

seconds_in_a_day = 24 * 60 * 60
seconds_in_a_day

seconds_in_a_week = 7 * seconds_in_a_day
seconds_in_a_week

import numpy as np
import IPython.display as display
from matplotlib import pyplot as plt
import io
import base64

ys = 200 + np.random.randn(100)
x = [x for x in range(len(ys))]

fig = plt.figure(figsize=(4, 3), facecolor='w')
plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)
plt.title("Sample Visualization", fontsize=10)

data = io.BytesIO()
plt.savefig(data)
image = F"data:image/png;base64,{base64.b64encode(data.getvalue()).decode()}"
alt = "Sample Visualization"
display.display(display.Markdown(F"""![{alt}]({image})"""))
plt.close(fig)
