import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

HOUSING_PATH = os.path.join("datasets", "housing")


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# importar os dados já armazenados
housing = load_housing_data()
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Apresentar uma tabela com amostra dos dados
print(housing.head())
# Apresenta informações importantes sobre os dados
print(housing.info())
# Apresentar contagem das informações dentro da coluna "ocean_proximity", por conter objetos
print(housing["ocean_proximity"].value_counts())
# Apresenta dados estatísticos sobre o conjunto
print(housing.describe())

# Apresenta um histograma
housing.hist(bins=50, figsize=(20, 15))
plt.show()

# Matriz de Correlação
corr_matrix = housing.corr()
# Apresenta as correlações do atributo "median_house_value" com os demais do conjunto
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Apresenta um conjunto de gráficos com a relação entre duas variáveis...
# É preciso escolher as variáveis...
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()

housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
plt.axis([0, 16, 0, 550000])
plt.show()

# Algumas manipulações
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

# Analisar mais uma vez a correlação, agora com as novas features
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Mais uma olhada nos dados
print(housing.describe())

# Extrair de housing a informação das labels, somente X
housing = strat_train_set.drop("median_house_value", axis=1)
# Criar um novo conjunto com as labels, somente y
housing_labels = strat_train_set["median_house_value"].copy()

# Exemplos de limpeza dos dados
# Verificiar valores incompletos
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
print(sample_incomplete_rows)

# podemos tomar algumas ações
# Opção 1 - drop
sample_incomplete_rows.dropna(subset=["total_bedrooms"])
# Opção 2 - drop
sample_incomplete_rows.drop("total_bedrooms", axis=1)
# Opção 3 - Preenchimento com valores
median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True)
print(sample_incomplete_rows)

# Tratando com atributos categóricos e textos
housing_cat = housing[['ocean_proximity']]
print(housing_cat.head(10))

# Codificando os atributos em valores numéricos
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:10])
print(ordinal_encoder.categories_)

# Outro método, usando representação binária
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot)
print(housing_cat_1hot.toarray())

