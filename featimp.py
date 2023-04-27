import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings("ignore")
rcParams['figure.figsize'] = 14, 7
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

def get_pearson_coef(df):
    print("\n\n### Get Feature Importances from Pearson Correlation coefficient ### \n")
    # Compute the correlation matrix
    corr = df.corr(method="pearson")
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    ax.set_title('Correlation and p-value - Pearson', size=20)
    plt.show()
    
def get_spearman_coef(df):
    print("\n\n### Get Feature Importances from Spearman Correlation coefficient ### \n")
    # Compute the correlation matrix
    corr = df.corr(method="spearman")
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    ax.set_title('Correlation and p-value - Spearman', size=20)
    plt.show()

def using_coef(df):
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    importances = pd.DataFrame(data={
        'Attribute': X_train.columns,
        'Importance': model.coef_[0]
    })
    importances = importances.sort_values(by='Importance', ascending=False)
    sns.barplot(importances['Attribute'], importances['Importance'], palette='coolwarm')
    plt.title('Feature importances from Logistic Regression coefficients', size=20)
    plt.xticks(rotation='vertical')
    plt.show()

def using_tree_based(df):
    model = XGBClassifier()
    model.fit(X_train_scaled, y_train)
    importances = pd.DataFrame(data={
        'Attribute': X_train.columns,
        'Importance': model.feature_importances_
    })
    importances = importances.sort_values(by='Importance', ascending=False)
    sns.barplot(importances['Attribute'], importances['Importance'], palette='coolwarm')
    plt.title('Feature importances from tree based coefficients', size=20)
    plt.xticks(rotation='vertical')
    plt.show()
    
def using_pca(df):
    pca = PCA().fit(X_train_scaled)
    loadings = pd.DataFrame(
        data=pca.components_.T * np.sqrt(pca.explained_variance_), 
        columns=[f'PC{i}' for i in range(1, len(X_train.columns) + 1)],
        index=X_train.columns
    )

    pc1_loadings = loadings.sort_values(by='PC1', ascending=False)[['PC1']]
    pc1_loadings = pc1_loadings.reset_index()
    pc1_loadings.columns = ['Attribute', 'CorrelationWithPC1']
    sns.barplot(pc1_loadings['Attribute'], pc1_loadings['CorrelationWithPC1'], palette='coolwarm')
    plt.title('PCA loading scores (first principal component)', size=20)
    plt.xticks(rotation='vertical')
    plt.show()



if __name__ == '__main__':
    # Load data
    data = load_wine()
    df = pd.concat([
        pd.DataFrame(data.data, columns=data.feature_names),
        pd.DataFrame(data.target, columns=['y'])
    ], axis=1)
    print("### Overview of the data ### \n") 
    df.info()
    X = df.drop('y', axis=1)
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.transform(X_test)
    # Get feature Importances using different techniques
    get_pearson_coef(df)
    get_spearman_coef(df)
    using_coef(df)
    using_tree_based(df)
    using_pca(df)