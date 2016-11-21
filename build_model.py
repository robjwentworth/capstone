import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier


# STEP 1
# ==============================================
# Load in the Auto Claims data from csv file state

df = pd.read_csv('data/auto_claim.csv')

# df is a pandas df
# Make X = our features

feature = df[['Gender', 'Income', 'Location Code', 'Monthly Premium Auto',
            'Months Since Last Claim', 'Number of Open Complaints', 'Vehicle Size']]

# STEP 2
# ==============================================
# Process our data


#Create Boolean 1/0 for Question is Customer have a Complaint? 1= Yes, 0 = No
feature['Complaint'] = feature['Number of Open Complaints'] != 0
feature.Complaint = feature.Complaint.astype(int)

#Convert objects to Categorical Data
feature.Gender = feature.Gender.astype("category")
feature['Location Code'] = feature['Location Code'].astype("category")
feature['Vehicle Size'] = feature['Vehicle Size'].astype("category")

#Get dummy variables for Categorical Data
df_dum_gen = pd.get_dummies(feature.Gender)
df_dum_loc = pd.get_dummies(feature['Location Code'])
df_dum_veh = pd.get_dummies(feature['Vehicle Size'])

#Combine all features into dataframe
df_all = pd.concat([df_dum_gen, df_dum_loc, df_dum_veh, feature.Income, feature['Monthly Premium Auto'],
            feature['Months Since Last Claim'],feature.Complaint], axis=1)

#Split features into X,y
v_features = df_all.columns
v_features = v_features.tolist()
v_features = v_features[:]
del v_features[v_features.index('Complaint')]

X = df_all.ix[:, v_features]
y = df_all.Complaint.astype('int')

# STEP 3
# ==============================================
# Fit our model
gbc = GradientBoostingClassifier(n_estimators=500, max_depth=13, subsample=0.5,
                                 max_features='auto', learning_rate=0.01)
gbc.fit(X, y)

# STEP 4
# ==============================================
# Export model and vectorizer to use it later

# Export our fitted model via pickle
with open('data/my_model.pkl', 'wb') as f:
    pickle.dump(gbc, f)
