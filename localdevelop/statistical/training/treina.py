import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr
import joblib
# from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    # Carrega os dados
    mydf = pd.read_csv('../../../datasets/statistical/BaseDefault01.csv')

    # Identifica no dataset as variáveis independentes e a variavel alvo
    independentcols = ['renda', 'idade', 'etnia', 'sexo', 'casapropria', 'outrasrendas', 'estadocivil', 'escolaridade']
    targetcol = 'default'
    x = mydf[independentcols]
    y = mydf[targetcol]

    # Cria o Classifier (modelo 1)
    clf = rfc()
    clf.fit(X=x, y=y)
    clf.independentcols = independentcols
    clf_acuracia = clf.score(X=x, y=y)
    print("Modelo 01 (classificador), criado com acurácia de: [{0}]".format(clf_acuracia))
    joblib.dump(clf, '../../../datasets/statistical/modelo01.joblib')
    print("Modelo 01 (classificador) salvo com sucesso.")


    # Cria o Regressor (modelo 2)
    independentcols = ['renda', 'idade', 'sexo', 'casapropria', 'outrasrendas', 'estadocivil', 'escolaridade']
    rgs = rfr()
    rgs.fit(X=x, y=y)
    rgs.independentcols = independentcols
    rgs_acuracia = rgs.score(X=x, y=y)
    print("Modelo 02 (Regressor), criado com acurácia de: [{0}]".format(rgs_acuracia))

    # Salva ambos os modelos

    joblib.dump(rgs, '../../../datasets/statistical/modelo02.joblib')
    print("Modelo 02 (regressor) salvo com sucesso.")
    pass
