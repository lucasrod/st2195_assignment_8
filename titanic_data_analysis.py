# ## Load necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the style of the visualizations
sns.set_style("whitegrid")

# ## Read the dataset
df = pd.read_csv("data/titanic.csv")

# Convert columns to appropriate data types
df['Sex'] = df['Sex'].map({'female': 'Female', 'male': 'Male'}).astype('category')
df['Survived'] = df['Survived'].map({0: 'No', 1: 'Yes'}).astype('category')
df['Pclass'] = df['Pclass'].map({1: 'First Class', 2: 'Second Class', 3: 'Third Class'}).astype('category')

# ## Question 1: Bar charts to describe the gender, ticket class, and survival of the passengers
# ### Bar chart for gender
plt.figure(figsize=(8, 6))
sns.countplot(x='Sex', data=df)
plt.title('Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# ### Bar chart for ticket class
plt.figure(figsize=(8, 6))
sns.countplot(x='Pclass', data=df)
plt.title('Ticket Class')
plt.xlabel('Ticket Class')
plt.ylabel('Count')
plt.show()

# ### Bar chart for survival status
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', data=df)
plt.title('Survival Rate')
plt.xlabel('Survival Status')
plt.ylabel('Count')
plt.show()

# ## Question 2: Passengers' age analysis by ticket class and survival status
# ### Histogram for passengers' age
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=16, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Summarizing the Age column
age_summary = df['Age'].describe()

# Converting the summary to a DataFrame
age_summary_df = age_summary.reset_index()

# Renaming the columns
age_summary_df.columns = ['Statistic', 'Value']

# Printing the summary as a table
print(age_summary_df)

# ### Boxplot for age per Ticket Class
plt.figure(figsize=(10, 8))
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Age by Ticket Class')
plt.xlabel('Ticket Class')
plt.ylabel('Age')
plt.show()

# Grouping by ticket class and summarizing age, including the number of NAs
age_ticket_class_summary = df.groupby('Pclass')['Age'].describe()
age_ticket_class_summary['Non_NAs'] = df.groupby('Pclass')['Age'].count()
age_ticket_class_summary['NAs'] = df.groupby('Pclass')['Age'].apply(lambda x: x.isnull().sum())

# Printing the summary as a table
print(age_ticket_class_summary)

# ### Boxplot for age per Survival status
plt.figure(figsize=(10, 8))
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age by Survival Status')
plt.xlabel('Survival Status')
plt.ylabel('Age')
plt.show()

# Grouping by survival status and summarizing age
age_survival_summary = df.groupby('Survived')['Age'].describe()
age_survival_summary['Non_NAs'] = df.groupby('Survived')['Age'].count()
age_survival_summary['NAs'] = df.groupby('Survived')['Age'].apply(lambda x: x.isnull().sum())

# Printing the summary as a table
print(age_survival_summary)

# ## Question 3: Travel fares and free passengers
# ### Histogram for Travel Fare
plt.figure(figsize=(10, 6))
sns.histplot(df['Fare'].dropna(), bins=50, kde=True)
plt.title('Travel Fares')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show()

# Summarizing the Fare column
fare_summary = df['Fare'].describe()

# Converting the summary to a DataFrame
fare_summary_df = fare_summary.reset_index()

# Renaming the columns
fare_summary_df.columns = ['Statistic', 'Value']

# Printing the summary as a table
print(fare_summary_df)

# ### Table with passenger fare payment status
# Creating a column for payment status
df['Payment_Status'] = df['Fare'].apply(lambda x: 'Did Not Pay' if x == 0 else 'Did Pay')

# Summarizing the payment status
payment_summary = df['Payment_Status'].value_counts().reset_index()
payment_summary.columns = ['Payment_Status', 'Count']

# Printing the summary as a table
print(payment_summary)

# ## Question 4: Different visualizations of family size per ticket class
# FamilySize is the sum of siblings/spouses (SibSp), parents/children (Parch), plus one for the passenger themselves
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Summarizing each column
sibsp_summary = df['SibSp'].describe()
parch_summary = df['Parch'].describe()
familysize_summary = df['FamilySize'].describe()

# Combining the summaries into a DataFrame
combined_summary_df = pd.DataFrame({
    'Siblings/Spouses': sibsp_summary,
    'Parents/Children': parch_summary,
    'Family Size': familysize_summary
})

# Transposing the DataFrame
combined_summary_df = combined_summary_df.transpose()

# Printing the combined summary as a table
print(combined_summary_df)

# Grouping by ticket class and describing family size
familysize_by_pclass_summary = df.groupby('Pclass')['FamilySize'].describe()

# Printing the summary as a table
print(familysize_by_pclass_summary)

# ## Vertical boxplot
plt.figure(figsize=(10, 8))
sns.boxplot(x='Pclass', y='FamilySize', data=df)
plt.title('Family Size by Ticket Class')
plt.xlabel('Ticket Class')
plt.ylabel('Family Size')
plt.show()

# ## Horizontal boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(y='Pclass', x='FamilySize', data=df, orient='h')
plt.title('Family Size by Ticket Class')
plt.ylabel('Ticket Class')
plt.xlabel('Family Size')
plt.show()

# ## Vertical histogram
g = sns.FacetGrid(df, col="Pclass", height=3, aspect=1)
g.map(sns.histplot, "FamilySize", bins=range(1, 12), kde=False)
g.set_titles("Family size in {col_name}")
g.set_axis_labels("Family size", "Count")
plt.show()

# ## Horizontal histogram
# In Python, horizontal histograms are not as straightforward as vertical ones.
# Instead, we use a barplot to achieve a similar effect.
for pclass in df['Pclass'].unique():
    subset = df[df['Pclass'] == pclass]
    plt.figure(figsize=(10, 6))
    sns.countplot(y='FamilySize', data=subset, order=range(1, subset['FamilySize'].max() + 1), color='steelblue')
    plt.title(f'Family Size by Ticket Class: {pclass}')
    plt.ylabel('Family Size')
    plt.xlabel('Count')
    plt.show()

# ## Vertical violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='Pclass', y='FamilySize', data=df)
plt.title('Family Size by Ticket Class')
plt.xlabel('Ticket Class')
plt.ylabel('Family Size')
plt.show()

# ## Horizontal violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(y='Pclass', x='FamilySize', data=df, orient='h')
plt.title('Family Size by Ticket Class')
plt.ylabel('Ticket Class')
plt.xlabel('Family Size')
plt.show()

# ## Question 5: Stacked bar charts to show survival rate by age and gender
# ### Stacked bar chart for survival by gender
survival_gender = pd.crosstab(df['Sex'], df['Survived'])

survival_gender.plot(kind='bar', stacked=True, figsize=(8, 6))
plt.title('Survival Count by Gender')
plt.xlabel('Gender')
plt.xticks(rotation=0)
plt.ylabel('Count')
plt.legend(title='Survived')
plt.show()

print(survival_gender)

# ### Stacked bar chart for survival by Ticket Class
survival_class = pd.crosstab(df['Pclass'], df['Survived'])

survival_class.plot(kind='bar', stacked=True, figsize=(8, 6))
plt.title('Survival Count by Ticket Class')
plt.xlabel('Ticket Class')
plt.xticks(rotation=0)
plt.ylabel('Count')
plt.legend(title='Survived')
plt.show()

# Printing the table
print(survival_class)

# ## Question 6: Violin plot to show survival rate by age and gender
# ### Violin plot for age distribution by survival and gender
plt.figure(figsize=(10, 6))
sns.violinplot(x='Sex', y='Age', hue='Survived', data=df, split=True)
plt.title('Age Distribution by Survival and Gender')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()

# ### Summary table of age by gender and survival status
# Group by gender and survival, then summarize the age distribution for each group
age_summary_by_group = df.groupby(['Sex', 'Survived'])['Age'].describe()

# Printing the summary table
print(age_summary_by_group)

# ## Question 7: Violin plot to show survival rate by age and ticket class
plt.figure(figsize=(10, 6))
sns.violinplot(x='Pclass', y='Age', hue='Survived', data=df, split=True)
plt.title('Age Distribution by Survival and Ticket Class')
plt.xlabel('Ticket Class')
plt.ylabel('Age')
plt.show()

# ### Summary table of age by ticket class and survival status
# Group by ticket class and survival, then summarize the age distribution for each group
age_summary_by_class = df.groupby(['Pclass', 'Survived'])['Age'].describe()

# Printing the summary table
print(age_summary_by_class)

