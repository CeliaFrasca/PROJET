#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 13:46:25 2024

@author: maison
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("PROJET TEMPÉRATURE TERRESTRE")
st.sidebar.image("/Users/maison/Desktop/climate-change-2254711_1280.png", use_column_width=True)
st.sidebar.title("Constat du dérèglement climatique : ")
pages=["Introduction","Au niveau mondial "," Dans l\'hémisphère nord", "Dans l\'hémisphère sud", "Par zone", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0]:
    st.title('Introduction générale')
    from PIL import Image
    from numpy import asarray
    img = Image.open('/Users/maison/Desktop/earth-4669629_1280.png')
    numpydata = asarray(img)
    st.image(numpydata, use_column_width= True)
    
    st.write('L’étude que nous avons réalisée porte sur l’évolution des anomalies de températures et des émissions de CO2 de 1880 à 2023.')
    st.write('Le constat a été réalisé à partir des bases de données en provenance de la NASA pour les estimations de changement des températures et du référentiel OWID (Our World In Data) pour le CO2.')
    st.write('L’évolution des anomalies de températures a été analysée : ')
    st.write('-	 Au niveau mondial pour Zorica TODOROVIC')
    st.write('-	 Au sein de l\'hémisphère Nord pour Célia FRASCA')
    st.write('-	 Au sein de l\'hémisphère Sud pour Rayan RIZQI' )
    st.write('-	 Par zones géographiques pour Nathan ZAMBELLI')
    st.write('Chaque partie comprend - une exploration des données - la modélisation - les prédictions avec une visualisation pour chaque partie.')


if page == pages[1]:
    st.header('AU NIVEAU MONDIAL')
    from PIL import Image
    from numpy import asarray
    img = Image.open('/Users/maison/Desktop/zorica.png')
    numpydata = asarray(img)
    st.image(numpydata, use_column_width= True)
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import streamlit as st
    import joblib 
    from scipy.stats import norm
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error, r2_score

    st.write("### CONTEXTE DU PROJET")
    
    st.write("Ce projet s’inscrit dans le constat de l’évolution des anomalies de températures depuis 1880 à 2023.")
    st.write("L’analyse a été réalisée à partir des variables 'années’ et 'moyenne annuelle'")
    st.write('')
    st.write('')
    st.write('')

    if st.checkbox('EXPLORATION DES DONNEES'):
        st.write("## 1- Exploration des données")
   
        st.write("#### a) Pré processing")
        df_temperature_2 = pd.read_csv('/Users/maison/Documents/PROJET/temperature_2.csv') 
        st.write(df_temperature_2.head())
    
        if st.checkbox("Afficher les valeurs manquantes") :
            st.write(df_temperature_2.isnull().sum())
        if st.checkbox("Afficher les doublons") :
            st.write(df_temperature_2.duplicated().sum())


        plt.figure(figsize=(12, 6))
        

        st.write("#### b) Visualisation de l'évolution des anomalies de températures")
        
        sns.lineplot(x='Year', y='J-D', data=df_temperature_2, marker='o')
        plt.title('Anomalies de Température Mondiale (J-D) de 1880 à 2023')
        plt.xlabel('Année')
        plt.ylabel('Anomalie de Température (°C)')
        plt.grid(True)
        st.pyplot(plt)   


        st.write("Durant la période de 1880 à 1940, les anomalies de température se maintiennent majoritairement en dessous de zéro, avec des conditions plus froides que la moyenne.")
        st.write("Un basculement vers le positif débute à partir de 1930 avec un pic de +0,20°C en 1940")
        st.write("À partir des années 1970, un changement de dynamique est perceptible. Les anomalies de température augmentent de manière régulière, avec une croissance de +0,25°C tous les 20 ans.")
 
    
        st.write("#### c) Distribution des températures annuelles")
    

        temperatures_annuelles = df_temperature_2['J-D']

        plt.figure(figsize=(10, 6))
        plt.hist(temperatures_annuelles, bins=20, color='skyblue', edgecolor='black')
        
        plt.xlabel('Température (°C)')
        plt.ylabel('Fréquence')
        plt.title('Distribution des températures annuelles')
        
        st.pyplot(plt) 
        
        st.write('On peut constater que la fréquence la plus élevée de distribution des anomalies de températures se situe entre -0,20°C et -0.10°C avec 24 années et la plus basse ou inexistante se situe entre 0,75°C et 0,80°C.')

    
        st.markdown("***Disribution avec la loi normale***")
        
        mean_temp = df_temperature_2['J-D'].mean()
        std_temp = df_temperature_2['J-D'].std()
        st.write("Moyenne des températures annuelles :", mean_temp)
        st.write("Écart-type des températures annuelles :", std_temp)


        mean_temp = 0.06770833333333333
        std_temp = 0.37863523556107614 
        num_samples = 1000
        simulated_temps = np.random.normal(mean_temp, std_temp, num_samples)

        x = np.linspace(mean_temp - 4*std_temp, mean_temp + 4*std_temp, 100)
        y = norm.pdf(x, mean_temp, std_temp)

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label='Loi normale', color='blue')
        plt.fill_between(x, y, color='lightblue', alpha=0.5)  
        plt.axvline(x=mean_temp, color='red', linestyle='--', label='Moyenne')

        plt.xlabel('Température (J-D)')
        plt.ylabel('Fréquence')
        plt.title('Distribution des températures annuelles avec loi normale')
        plt.legend()
        plt.grid(True)

        st.pyplot(plt)

        st.write("La zone sous la courbe normale est remplie en bleu clair et illustre la distribution des anomalies des températures.")
        st.write("On constate que la plus grande distribution se situe entre -0,25°C et 0,40°C.")

    
        st.write("#### d) Comparatif de la distribution des anomalies de températures avant et après 1950 avec la loi normale")
     
      
        num_samples = 1000
        mean_temp_periode1 = -0.19971830985915492
        std_temp_periode1 = 0.1528208272173277

        mean_temp_periode2 = 0.32780821917808217
        std_temp_periode2 = 0.35059570961176 

        simulated_temps = np.random.normal(mean_temp, std_temp, num_samples)
        x=df_temperature_2['Year']
        y=df_temperature_2['J-D']

        periode1 = df_temperature_2[(df_temperature_2['Year'] >= 1880) & (df_temperature_2['Year'] <= 1950)]
        periode2 = df_temperature_2[(df_temperature_2['Year'] >= 1951) & (df_temperature_2['Year'] <= 2023)]

        moyenne_periode1 = periode1['J-D'].mean()
        ecart_type_periode1 = periode1['J-D'].std()

        moyenne_periode2 = periode2['J-D'].mean()
        ecart_type_periode2 = periode2['J-D'].std()

        st.write("Période1 (1880-1950) - Moyenne:", moyenne_periode1, "Écart-type:", ecart_type_periode1)
        st.write("Période2 (1951-2023) - Moyenne:", moyenne_periode2, "Écart-type:", ecart_type_periode2)

        periode1 = df_temperature_2[(df_temperature_2['Year'] >= 1880) & (df_temperature_2['Year'] <= 1950)]
        periode2 = df_temperature_2[(df_temperature_2['Year'] >= 1951) & (df_temperature_2['Year'] <= 2023)]

        plt.subplot(2, 2, 1)
        x = np.linspace(mean_temp_periode1 - 3*std_temp_periode1, mean_temp_periode1 + 3*std_temp_periode1, 100)
        y = norm.pdf(x, mean_temp_periode1, std_temp_periode1)
        plt.hist(periode1['J-D'], bins=30, density=True, alpha=0.6, color='g')
        plt.plot(x, y, color='blue')
        plt.axvline(x=mean_temp_periode1, color='red', linestyle='--')
        plt.title('Période 1 (1880-1950)')
        plt.xlabel('Température (J-D)')
        plt.ylabel('Fréquence')

    # Période 2 
        plt.subplot(2, 2, 2)
        plt.hist(periode2['J-D'], bins=30, density=True, alpha=0.6, color='g')
        x = np.linspace(mean_temp_periode2 - 3*std_temp_periode2, mean_temp_periode2 + 3*std_temp_periode2, 100)
        y = norm.pdf(x, mean_temp_periode2, std_temp_periode2)
        plt.plot(x, y, color='blue')
        plt.axvline(x=mean_temp_periode2, color='red', linestyle='--')
        plt.title('Période 2 (1951-2023)')
        plt.xlabel('Température (J-D)')
        plt.ylabel('Fréquence')

        st.pyplot(plt)

        st.write("On constate très vite, qu’avant   1950, la moyenne est en négative avec -0,20°C environ et un écart-type de 0,15°C. Après 1950, la moyenne passe en positive avec 0,32°C soit +0,50°C et un écart-type de 0,32°C soit le double.")

        periode_avant_1950 = (df_temperature_2['Year'] <= 1950)
        periode_apres_1950 = (df_temperature_2['Year'] > 1950)

        temperature_avant_1950 = df_temperature_2.loc[periode_avant_1950, 'J-D']
        temperature_apres_1950 = df_temperature_2.loc[periode_apres_1950, 'J-D']

        stats_avant_1950 = temperature_avant_1950.describe()
        stats_apres_1950 = temperature_apres_1950.describe()

     
        plt.figure(figsize=(8, 6))
        plt.boxplot([temperature_avant_1950, temperature_apres_1950], labels=['Avant 1950', 'Après 1950'], patch_artist=True, showmeans=True)

    # Ajout des étiquettes et des titres
        plt.xlabel('Période')
        plt.ylabel('Température (°C)')
        plt.title('Comparaison des températures avant et après 1950')
        st.pyplot(plt)


        
    # SUITE ZORICA 
    
    
    
    if st.checkbox('MODELISATION DES DONNEES'):
        st.write("##  2- Modélisation des données")

        st.write("Dans cette modélisation trois modèles différents sont utilisés et comparés entre eux pour modéliser et prédire l'évolution des anomalies de température.")
        st.write("")

        df_temperature_2 = pd.read_csv('/Users/maison/Documents/PROJET/temperature_2.csv')  
  

        linear_regression_metrics = {
           'Modèle': 'Régression Linéaire',
           'Score sur ensemble train': 0.7623622231878608,
           'Score sur ensemble test': 0.7655299070041347,
           'Mean Squared Error': 0.021687061726404964,
           'R^2 Score': 0.7655299070041347,
           'R^2_poly Score': 0.9082844953219175,
           }

        decision_tree_metrics = {
           'Modèle': 'Decision Tree Regressor',
           'Score sur ensemble train': 1.0,
           'Score sur ensemble test': 0.8540444853536691,
           'Mean Squared Error': 0.0135,
           'R^2 Score': 0.8540444853536691,
           'R^2_poly Score': 1.0,
           }

        random_forest_metrics = {
           'Modèle': 'Random Forest',
           'Score sur ensemble train': 0.9913418205103421,
           'Score sur ensemble test': 0.9011774169081367,
           'Mean Squared Error': 0.009140489655172421,
           'R^2 Score': 0.9011774169081367,
           'R^2_poly Score': 0.9913045266072046,
           }

        metrics_df = pd.DataFrame([linear_regression_metrics, decision_tree_metrics, random_forest_metrics])
        st.write(metrics_df)
   
        X = df_temperature_2['Year'].values.reshape(-1, 1)
        y = df_temperature_2['J-D'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       
        regressor = LinearRegression()
        regressor.fit(X, y)
   
        y_pred = regressor.predict(X)

        poly_reg = PolynomialFeatures(degree = 10)
        X_poly = poly_reg.fit_transform(X)
        regressor_poly = LinearRegression()
        regressor_poly.fit(X_poly, y)
        y_poly_pred = regressor_poly.predict(X_poly)

        plt.scatter(X, y, color ="yellow")
        plt.plot (X, regressor.predict(X), color = 'red')
        plt.scatter(X, y, color = 'green')
        plt.plot(X, regressor.predict (X), color ='red')
        plt.plot(X, regressor_poly.predict(X_poly), color = 'BLUE')
        st.pyplot(plt)

        st.write('On peut observer une courbe plus ajustée aux données de températures par rapport à la droite linéaire mais beaucoup de données restent encore perdues.')

        poly_reg = PolynomialFeatures(degree = 10)
        X_poly = poly_reg.fit_transform(X)
        regressor_poly = DecisionTreeRegressor()
        regressor_poly.fit(X_poly, y)
        y_poly_pred = regressor_poly.predict(X_poly)

        plt.scatter(X, y, color = 'green')
        plt.plot(X, regressor.predict (X), color ='red')
        plt.plot(X, regressor_poly.predict(X_poly), color = 'blue')
        st.pyplot(plt)

        st.write('On constate que la courbe est ajustée parfaitement aux données et qu’elles sont quasiment toutes prise en compte.')
       
        poly_reg = PolynomialFeatures(degree = 10)
        X_poly = poly_reg.fit_transform(X)
        regressor_poly = RandomForestRegressor()
        regressor_poly.fit(X_poly, y)
        y_poly_pred = regressor_poly.predict(X_poly)

        plt.scatter(X, y, color = 'green')
        plt.plot(X, regressor.predict (X), color ='red')
        plt.plot(X, regressor_poly.predict(X_poly), color = 'blue')
        st.pyplot(plt)
       
        st.write('La courbe est très bien ajustée aux données. Cependant, quelques points ne sont pas pris en compte mais le pourcentage est très faible.')
        
        
        
        
        
        
        
        
        
        
    if st.checkbox('PREDICTIONS'):
        st.write("## 3- Prédictions")

        df_temperature_2 = pd.read_csv('/Users/maison/Documents/PROJET/temperature_2.csv')  
   
        
    
        years = df_temperature_2['Year'].values.reshape(-1, 1)
        temperatures = df_temperature_2['J-D'].values

        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(years)

        model = LinearRegression()
        model.fit(X_poly, temperatures)

        future_years = np.array([[2025], [2026], [2027], [2028], [2029], [2030], [2040], [2050],[2100]])
        future_years_poly = poly_features.transform(future_years)

        predicted_temperatures_future = model.predict(future_years_poly)

        st.write("Prévisions des températures pour les années 2025 à 2100 :")
        
   
   
        years_historical = df_temperature_2['Year']
        temperatures_historical = df_temperature_2['J-D']

        years_future = np.array([[2025], [2026], [2027], [2028], [2029], [2030], [2040], [2050],[2100]])
        temperatures_future = predicted_temperatures_future

        all_years = np.concatenate([years_historical, years_future.flatten()])
        all_temperatures = np.concatenate([temperatures_historical, temperatures_future])
        plt.hist2d(all_years, all_temperatures, bins=(30, 30), cmap='Blues')
        plt.xlabel('Année')
        plt.ylabel('Température (°C)')
        plt.title('Évolution des températures (1880-2023) avec prévisions pour 2025-2100')

        plt.legend()
       
        st.pyplot(plt)

        years = np.array([2025, 2026, 2027, 2028, 2029, 2030, 2040, 2050, 2100])
        temperatures = predicted_temperatures_future   
    
        df = pd.DataFrame({'Années': years, 'Températures': temperatures})
        st.table(df)

        st.write('On peut observer une constante évolution :')
        st.write ('- +0,02 °C par an')
        st.write('- +0,20°C tous les 10 ans')
        st.write('- +1,50°C tous les 50 ans')

        st.write('CONCLUSION')
        st.write('Les modèles utilisés permettent dans un premier temps de comprendre l’évolution des températures et d’en rechercher la cause de l’évolution.')
        st.write('Ces modèles permettent aussi de prédire les évolutions dans les années futures et à partir des données historiques. ')


if page == pages[2]:
        
    st.header('DANS L\'HÉMISPHÈRE NORD')
    from PIL import Image
    from numpy import asarray
    img = Image.open('/Users/maison/Desktop/polar-bear-5906016_1280.png')
    numpydata = asarray(img)
    st.image(numpydata)
    
    st.write('L’objectif ici est de faire un constat du dérèglement climatique dans l’hémisphère Nord en établissant des comparaisons avec des périodes antérieures à notre époque.')
    st.write('')
    st.write('')
    st.write('')
    if st.checkbox('EXPLORATION DES DONNÉES'):
        st.header('1 - Exploration des données')
        
        df=pd.read_csv("/Users/maison/Documents/PROJET/NH.Ts+dSST(3).csv", header = 1)
        co2 = pd.read_csv('/Users/maison/Documents/PROJET/owid-co2-data(1).csv')
        st.write('le travail a été réalisé sur La base de données de la NASA  - Goddard Institute for space studies (GISS) relative aux estimations globales de changements de températures de surface dans l\'hémisphère Nord de 1880 à nos jours') 
        st.write('Cette base de données regroupe les anomalies de températures par rapport à la moyenne des températures relevées entre 1951 et 1980.')
        st.write('Ce dataset  décompose les anomalies de températures par années selon les mois, l\'année civile, et selon l\'année et les saisons météorologiques : Decembre-Janvier-Fevrier : Hiver, Mars-Avril-Mai : printemps, Juin-Juillet-Août : été et Septembre-Octobre-Novembre : automne')
        
        if st.checkbox('Afficher le dataframe de la NASA',):
            st.dataframe(df.head())
        
        st.write('L\'analyse a également été réalisée sur l\'étude de la base de données Data on CO2 and Greenhouse Gas Emissions’ de ‘Our world in data’, une organisation à but non lucratif qui met à disposition des bases de données dont les données sources proviennent de différentes entités (pour le dataset utilisé : Energy Institute (EI), U.S. Energy Information Administration (EIA), Global Carbon Project, Jones et al. , Climate Watch, University of Groningen GGDC\'s Maddison Project Database, Bolt and van Zanden).')
        if st.checkbox('Afficher le dataframe de OUR WOLD IN DATA'):
                st.dataframe(co2.head(10))
        st.write('Pour répondre à l’objectif du projet, à savoir faire un constat du dérèglement climatique dans l’hémisphère Nord, ce sont les données du dataset de la NASA qui ont été principalement utilisées. En effet, pour constater le dérèglement climatique, il faut en regarder ses effets, et l’anomalie des températures en est un des principaux.')
        st.write('En revanche, et dans le but d’aller plus loin dans la démarche, il paraissait important de mettre en corrélation ces anomalies avec l’un des facteurs principaux du dérèglement climatique : les émissions de CO2.')

    
        st.header('les données sur les anomalies de températures')
        st.write('Pour plus de compréhension des données à manipuler, les colonnes ont été renommées, et les colonnes inutiles supprimées.')
        st.write('')
        st.write('Concernant la qualité des données, seules 2 valeurs manquantes étaient présentes dans le dataframe. Les colonnes concernées étant utilisées principalement pour des calculs de sommes ou de différences, il a été choisi de les remplacer par la valeur 0 qui n’impactera pas les calculs, et permettra de conserver les autres données de la ligne.')
        st.write('')
        st.write('Des outliers ont été  détectés.')
        st.write('')
     
        df = df.replace('***', '0')
        df['DJF'] = df['DJF'].astype(float)
        df['J-D'] = df['J-D'].astype(float)
        df = df.rename(columns = {'J-D' : 'Année_civile', 
                               'DJF' : 'hiver', 
                               'JJA': 'été'})
        df = df.drop(['SON', 'MAM', 'D-N'], axis = 1)
    
        fig = plt.figure()
        sns.boxplot(df['Année_civile'])
        plt.title('boxplot des Anomalies de températures')
        st.pyplot(fig)
     
        if st.checkbox('Regardons à quoi correspondent ces valeurs : '):
            st.write('')
            st.dataframe(df.loc[df['Année_civile'] >= 0.9])
        
     
        st.write(' Il ne s’agit pas de valeurs aberrantes mais bien de valeurs extrêmes sur les années les plus récentes. Ces données sont importantes pour notre analyse, elles sont donc à conserver')
     
     
        st.header('Les données sur le CO2')
        st.write('')
        st.write('Ici, il est question de récupérer les données concernant les émissions de CO2 au niveau mondial afin de les intégrer à notre dataframe relatif aux anomalies de températures')
        st.write('')
        st.write('Il est donc nécéssaire de filtrer le dataset pour :')
        st.write('      - Ne retenir que les années à partir de 1880')
        st.write('      - Ne retenir que les émissions concernant le monde (country = World)')
        st.write('')
        st.write('Nous retenons la colonne CO2 including luc qui nous permet d\'avoir les émissions de CO2 les plus complètes')
        co2 = co2[['year','country' ,'co2_including_luc']]
        co2 = co2.loc[co2['year']>= 1880]
        co2 = co2.loc[co2['country'] == 'World']
        co2 = co2.drop('country', axis = 1)
        co2 = co2.rename(columns = {'co2_including_luc' : 'CO2'})
        st.dataframe(co2.head())
        st.write('Dans ce dataframe nous n\'avons aucune donnée manquante , aucun doublon et aucun outlier')
        st.write('')
        st.write('Nous fusionnons nos dataframes et obtenons le jeu de données suivant : ')
        df = df.rename(columns = {'Year': 'year'})
        data = pd.merge(df, co2, on = 'year')
        st.dataframe(data.head(10))
    
    
    
    if st.checkbox('DATA VISUALISATION'):
        st.header('DATAVISUALISATION')
        st.write('')
        st.subheader('1 - Evolution des anomalies de températures')
        st.write('')
        st.write('')
    

        import plotly.express as px
        df=pd.read_csv("/Users/maison/Documents/PROJET/NH.Ts+dSST(3).csv", header = 1)
        co2 = pd.read_csv('/Users/maison/Documents/PROJET/owid-co2-data(1).csv') 
        co2 = co2[['year','country' ,'co2_including_luc']]
        co2 = co2.loc[co2['year']>= 1880]
        co2 = co2.loc[co2['country'] == 'World']
        co2 = co2.drop('country', axis = 1)
        co2 = co2.rename(columns = {'co2_including_luc' : 'CO2'})
        df = df.rename(columns = {'Year': 'year',
                               'J-D' : 'Anomalies °C'})
        data = pd.merge(df, co2, on = 'year')    
        data = data.drop('SON', axis = 1) 
        data = data.drop(['D-N', 'MAM'], axis = 1)
        data = data.rename(columns = {'JJA': 'été',
                                  'DJF' : 'hiver'})
    
   

        var_x = 'year'
        var_y = data['Anomalies °C']
        fig3 = px.scatter(
            data_frame = data,
            x = var_x,
            y = var_y,
            color = var_x, 
            title = 'EVOLUTION DES ANOMALIES DE TEMPERATURES ANNUELLES - HEMISPHERE NORD (1880-2023)')
       
        
        st.plotly_chart(fig3, theme = None)
    
        st.write('A l\'aide de ce graphique, il est aisé de constater l\'augmentation des anomalies de températures')
        st.write('Ce qui s’apparente à un “plateau” est en réalité l’intervalle de comparaison (1951-1980), les anomalies étant interprétées par rapport aux moyennes relevées entre 1951 et 1980')
        
        st.write('')
        st.write('')
        if st.button('EN SAVOIR PLUS SUR L\'ANOMALIE DES TEMPERATURES', use_container_width = True):
       
        
            st.write(' SUR LA PERIODE DE 1880-2023 :')
            st.write('      - Valeur maximale relevée :    +1,49 en 2023')
            st.write('      - Valeur minimale relevée :    -0.57 en 1917')
            st.write(' Nous avons donc une ETENDUE DE 2,06°C')
            st.write('')
            st.write('Portons l\'observation sur deux périodes identiques de 43 ans, se trouvant en dehors de la période de référence de la NASA  (1951 -1980) : ')
            st.write('      - Etendue de 1908 à 1951 :    +0,84°C')
            st.write('      - Etendue de 1980 à 2023 :    +1,35°C')
    
        st.write('Il est évident que l\'hémisphère nord subit un réchauffement climatique qui, de surcroît, à une tendance à s\'accélérer. Les valeurs extrêmes qui ont été détectées lors de l\'exploration des données en sont un fort indicateur.')
        st.write('')
    
  
        

        
        st.subheader('2 - Analyse des étés et des hivers')
        st.write('')
        st.write('Observons-nous une tendance au réchauffement sur les saisons ? ')
        st.write('Allons nous constater des hivers de plus en plus froids et des étés de plus en plus chaud ? Ou bien une tendance qui s\'inverse: des étés qui se refraîchissent et des hivers qui se réchauffent ?')
        st.subheader('Etude des périodes estivales')
        var_x = 'year'
        var_y = 'été'
        fig3 = px.scatter(
            data_frame = data,
            x = var_x,
            y = var_y,
            color = var_x, 
            title = 'EVOLUTION DES ANOMALIES DE TEMPERATURES ESTIVALES - HEMISPHERE NORD (1880-2023)')
    
        st.plotly_chart(fig3, theme = None)
    
        st.write('')
        st.write('A l’instar de la tendance annuelle, nous constatons que les étés sont de plus en plus chauds.')
        if st.button('En savoir plus sur l\'évolution des anomalies - été', use_container_width = True):
            st.write(' SUR LA PERIODE DE 1880-2023 :')
            st.write('     - Anomalie maximale relevée  : +1,57°C (en 2023)')
            st.write('     - Anomalie minimale relevée : -0,77°C (en  1912)')
            st.write ('nous avons donc une ETENDUE DE + 2.34 °C')
            st.write('')
            st.write('Portons l\'observation sur deux périodes identiques de 43 ans, se trouvant en dehors de la période de référence de la NASA  (1951 -1980) : ')
            st.write('      - Etendue de 1908 à 1951 :    +1,05°C')
            st.write('      - Etendue de 1980 à 2023 :    +1,30°C')

        
        st.write('')
        st.write('Nous pouvons de nouveau mettre en exergue que les étés se réchauffent et que la tendance est également à l\'accélération' )
   
    
        st.subheader('Etude des périodes hivernales')
        var_x = 'year'
        var_y = 'hiver'
        fig3 = px.scatter(
            data_frame = data,
            x = var_x,
            y = var_y,
            color = var_x,
            title = 'EVOLUTION DES ANOMALIES DE TEMPERATURES HIVERNALES - HEMISPHERE NORD (1880-2023)')
  
        st.plotly_chart(fig3, theme = None)

  
    
        st.write('')
        st.write('Nous n\'assistons pas à des hivers de plus en plus froids, mais bien à des hivers qui se réchauffent.')
        if st.button('En savoir plus sur l\'évolution des anomalies - hiver', use_container_width = True):
        
            st.write(' SUR LA PERIODE DE 1880-2023 :')
            st.write('     - Anomalie maximale relevée  : +1,93°C (en 2016)')
            st.write('     - Anomalie minimale relevée : -1,50°C (en  1893)')
            st.write ('nous avons donc une ETENDUE DE + 3.43 °C')
            st.write('')
            st.write('Portons l\'observation sur deux périodes identiques de 43 ans, se trouvant en dehors de la période de référence de la NASA  (1951 -1980) : ')
            st.write('      - Etendue de 1908 à 1951 :    +1,91°C')
            st.write('      - Etendue de 1980 à 2023 :    +2,35°C')
        

        st.write('Les hivers ont tendance à se réchauffer plus vite que les étés au regard des étendues calculées :')
        st.write('      - +1.30 °C sur les étés depuis 1980')
        st.write('      - +2.25 °C sur les hivers sur cette même période.')
    
    
    
        st.header('3 - Evolution des émissions de CO2')
        import plotly.express as px
   
    
        var_x = 'year'
        var_y = 'CO2'
        fig3 = px.bar(
            data_frame = data,
            x = var_x,
            y = var_y,
            color = var_y,
            title = 'EVOLUTION DES EMISSIONS MONDIALES DE CO2 (1880-2023)')
  
        st.plotly_chart(fig3, theme = None)

    
        st.write('')
        st.write('Concernant les émissions de CO2 mondiales, nous constatons une permanente évolution à la hausse. Nous notons une accélération depuis 1945 , l’après-seconde guerre mondiale constitue un tournant : les émissions dues aux combustibles fossiles (charbon, pétrole, gaz) deviennent majoritaires.')
        st.header('4 - Mise en relation CO2 et anomalies de températures')
        
        st.write('Rappelons-nous l\'évolution de nos variables :' )
        var_a_choisir =['CO2', 'Anomalies °C']
        # nuage de points interactifs :
        var_y = st.selectbox ('CHOISIS TA VARIABLE', var_a_choisir)
        var_x = data['year']
        var_z = data['CO2']
        fig = px.scatter(
                data_frame = data,
                x = var_x,
                y = var_y,
                color = var_z )
         
        st.plotly_chart(fig, theme = None)

        st.write('Elles ont toutes les deux une évolution à la hausse avec de surcroît une accélération, regardons à présent leur corrélation ? Quel lien entretiennent-elles ?')
   
    
        var_x = 'CO2'
        var_y = 'Anomalies °C'
        var_z = 'year'
    
        fig4 = px.scatter(
                data_frame = data,
                x = var_x, 
                y= var_y,
                color = var_z,
                size = var_x, 
                trendline = 'ols', 
                title = 'RELATION ENTRE LES ANOMALIES °C ET LES EMISSIONS DE CO2')
    
        st.plotly_chart(fig4, theme = None)
    
    
    
        st.write('')
        st.write('Nous constatons une relation entre les émissions de CO2 et les anomalies de températures : Plus les émissions de CO2 sont importantes et plus les anomalies de températures augmentent.')
        st.write('Nous constatons également que ce sont bien les années les plus récentes qui sont présentes dans les émissions de CO2 et dans les anomalies de températures les plus élevées. La linéarité apparaît clairement à partir des années 1980.')
        if st.button('COEFFICIENT DE PEARSON', use_container_width = True):
            st.write('Vérifions la corrélation à l’aide du test de corrélation de Pearson :')
            st.write('     - Hypothèse nulle : les variables ‘émissions de CO2’ et ‘anomalies de températures’ ne sont pas corrélées')
            st.write('     - Hypothèse alternative : les variables sont corrélées.')
            st.write('')
            st.write('Sur la période de 1880 à 2023, la P Valeur est de 2.76, nous pouvons donc conclure que les variables sont bien corrélées. A quel point le sont-elles?')
            st.write('Le coefficient de corrélation est de 0.90 ce qui nous informe que les variables sont fortement corrélées.')
            st.write('Observons si l’intensité de cette relation augmente depuis 1980:')
            st.write('Pour cela nous séparons notre jeu de données en 2 séquences identiques (nous excluons la période de référence de la NASA 1951-1980)')
            st.write('   - Première séquence : 1908 -1951 : le coefficient est de 0.73')
            st.write('   - Deuxième séquence : 1980-2023 : le coefficient est de 0.91')
            st.write('La corrélation s\'intensifie')
        
        df=pd.read_csv("/Users/maison/Documents/PROJET/NH.Ts+dSST(3).csv", header = 1)
        co2 = pd.read_csv('/Users/maison/Documents/PROJET/owid-co2-data(1).csv') 
        co2 = co2[['year','country' ,'co2_including_luc']]
        co2 = co2.loc[co2['year']>= 1880]
        co2 = co2.loc[co2['country'] == 'World']
        co2 = co2.drop('country', axis = 1)
        co2 = co2.rename(columns = {'co2_including_luc' : 'CO2'})
        df = df.rename(columns = {'Year': 'year',
                               'J-D' : 'Anomalies °C'})
        data = pd.merge(df, co2, on = 'year')    
        data = data.drop('SON', axis = 1) 
        data = data.drop(['D-N', 'DJF', 'MAM', 'JJA'], axis = 1)
     
     


             
             
             
        
    
    
    
    if st.checkbox('MODÉLISATION'):
        st.header('MODÉLISATION')
        st.write('')
        st.subheader('1 - Objectif et méthodologie')
        st.write('')
        st.write('')
        st.write('Notre modèle de machine learning s’utiliserait à des fins de prédictions concernant les anomalies de températures sur les années à venir concernant l’hémisphère Nord.')
        st.write('Notre variable cible (celle que l’on cherche à prédire) est amenée à prendre des valeurs numériques continues, en conséquence, nous sommes en présence d\' un problème de régression en apprentissage supervisé puisque l’objectif est de créer un modèle qui serait en mesure de généraliser une situation apprise.')
        st.write('La taille d’échantillon test de l\'ordre de 20%  du jeu de données pour l\'entraînement des modèles. Pour chaque modèle nous conserverons la même taille d‘échantillon test.')
        st.write('Nous séparons le jeu de données en isolant la variable cible :  “Anomalies °C” , des variables explicatives :  “Année” et “ CO2”. ')
        st.write('Concernant l’hémisphère Nord, il s’agira de trouver un modèle qui sache gérer les valeurs extrêmes positives du fait des anomalies de températures s\'accélèrant à la hausse dans cet hémisphère depuis 2015')
        st.write('')
        df=pd.read_csv("/Users/maison/Documents/PROJET/NH.Ts+dSST(3).csv", header = 1)
        co2 = pd.read_csv('/Users/maison/Documents/PROJET/owid-co2-data(1).csv') 
        co2 = co2[['year','country' ,'co2_including_luc']]
        co2 = co2.loc[co2['year']>= 1880]
        co2 = co2.loc[co2['country'] == 'World']
        co2 = co2.drop('country', axis = 1)
        co2 = co2.rename(columns = {'co2_including_luc' : 'CO2'})
        df = df.rename(columns = {'Year': 'year',
                                  'J-D' : 'Anomalies °C'})
        data = pd.merge(df, co2, on = 'year')    
        data = data.drop('SON', axis = 1) 
        data = data.drop(['D-N', 'DJF', 'MAM', 'JJA'], axis = 1)
        data_modelisation = data.drop(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], axis = 1)
    
    
        if st.checkbox('Voir le jeu de données'):
            st.dataframe(data_modelisation)
            if st.checkbox('Voir les corrélations de nos variables'):
                        correlation = data_modelisation.corr()
                        fig0 = plt.figure()
                        sns.heatmap(correlation, annot = True, cmap = 'crest')
                        st.pyplot(fig0)
        
        data_modelisation['Anomalies °C'] = data_modelisation['Anomalies °C'].astype(float)
        st.header('2 - Les Modèles')
   
        
        Modeles = st.selectbox(label = 'Modèles pour la variable explicative \'Année\'', options = ['Regression lineaire', 'Decision tree', 'Random Forest', 'Lasso', 'Polynomial'], index=None, placeholder="Selectionne un modèle..." )
        Modeles_CO2 = st.selectbox(label = 'Modèles pour le CO2', options = ['Regression lineaire', 'Decision tree', 'Random Forest', 'polynomial'], index=None, placeholder="Selectionne un modèle..." )

        if Modeles == 'Regression lineaire': 
            st.header('La régression linéaire')
            st.subheader('Préparation du jeu de données')
            st.write('Le dataset n\'a que des variables numériques mais d\'unités de mesure différentes,  il est nécessaire de procéder à leur transformation afin de les rendre comparables. Nos variables ne suivent pas une loi normale donc nous aurions pu nous orienter vers une normalisation mais, au regard de la présence d’outliers, nous opterons pour un Robust Scaler qui est moins sensible aux valeurs extrêmes.')
            #entrainement du modele 
            feats = data_modelisation['year'].values.reshape(-1,1)
            target = data_modelisation['Anomalies °C']
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state = 0)
    
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            
            from sklearn.linear_model import LinearRegression
            model_lr = LinearRegression()
            model_lr.fit(X_train_scaled, y_train)
            
            st.subheader('Résultats du modèle')
            st.write('Le score sur le jeu d\'entrainement est de :', model_lr.score(X_train_scaled, y_train))
            st.write('Le score sur le jeu de test est de:', model_lr.score(X_test_scaled, y_test))
            st.write('Ici, nous avons 73% des données qui sont expliquées par le modèle sur le jeu d\'entrainement et  71% sur le jeu de test.')
            st.write('')
            st.subheader('Analyse des prédictions')
            score_train_lr = model_lr.score(X_train_scaled, y_train)
            score_test_lr = model_lr.score(X_test_scaled, y_test)
            
            y_pred_lr = model_lr.predict(X_test_scaled)
           
            import plotly.express as px
            
            
            
    # graphique predictions lr
            feats_scaled = scaler.fit_transform(feats)
            predictions = model_lr.predict(feats_scaled)
            
            import plotly.graph_objects as go
            fig43 = go.Figure()
            fig43.add_trace(go.Scatter(x=data_modelisation['year'], y=target,
                                mode='markers', name='Données réelles', marker=dict(color='cornflowerblue')))
            fig43.add_trace(go.Scatter(x=data_modelisation['year'], y=model_lr.predict(feats_scaled),
                                mode='lines', name='Prédictions', line=dict(color='red')))

            fig43.update_layout(title='Prédictions de la regression linéaire',
                        xaxis_title='Années',
                        yaxis_title='Anomalies °C',
                        showlegend=True,
                        template='plotly_white')

            st.plotly_chart(fig43)



            
            
            st.write('')
            st.subheader('Les résidus')
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            st.write('La MAE (Somme des valeurs absolues des écarts des erreurs) est de :', mean_absolute_error(y_test, y_pred_lr))
            st.write('Le MSE (moyenne des distances entre les vraies valeurs et les valeurs prédites, élevées au carré) est de :', mean_squared_error(y_test, y_pred_lr))
            st.write('La RMSE (Racine carrée de la MSE) est de :', mean_squared_error(y_test, y_pred_lr, squared = False))
            st.write('')
            
            mae_lr = mean_absolute_error(y_test, y_pred_lr)
            mse_lr = mean_squared_error(y_test, y_pred_lr)
            rmse_lr = mean_squared_error(y_test, y_pred_lr, squared = False)
            
# graphique sur les erreurs de predictions  
            
            erreurs_predictions_lr = model_lr.predict(feats_scaled)-target
            erreur_pred_lr = pd.DataFrame({'realité' : target, 
                                            'predictions' : model_lr.predict(feats_scaled), 
                                            'erreur de prediction' : erreurs_predictions_lr})
            erreur_pred_lr['qualite prediction']= pd.cut(erreurs_predictions_lr, bins =[ -0.8,-0.10,-0.02,0.02,0.10, 0.8], labels = ['mauvaise (-)', 'tolérée (-)', 'parfaite', 'tolérée (+)', 'mauvaise(+)'])
            import plotly.express as px
            
            var_x = 'realité'
            var_z = 'erreur de prediction'
            var_y = 'predictions'
            var_t = 'qualite prediction'
          
            fig51 = px.scatter(
                data_frame = erreur_pred_lr,
                x = var_x, 
                y= var_y,
                color = var_z,
                
                title = 'Resultat des predictions et ses erreurs')
            st.plotly_chart(fig51, theme = None)
            
            
# countplot residus 

            
             
            st.write('Ici nous considérons une erreur de prédiction entre -0.02°C et 0.02°C comme parfaite, et nous tolérons les erreurs inférieures à 0,10 °C, considérant toutes les autres erreurs comme mauvaises')
            fig100= plt.figure()
            sns.countplot(erreur_pred_lr, x= 'qualite prediction', color = 'cornflowerblue')
            sns.color_palette('flare', as_cmap = True)
            st.pyplot(fig100)
            
            
            
            
        if Modeles == 'Decision tree': 
            st.header('L\'arbre de décision')
            # entrainement du modele 
            feats = data_modelisation[ 'year'].values.reshape(-1,1)
            target = data_modelisation['Anomalies °C']
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state = 0)
            from sklearn.tree import DecisionTreeRegressor
            model_dtr = DecisionTreeRegressor()
            model_dtr.fit(X_train, y_train)
            st.subheader('Résultats du modèle')
            st.write('Le score sur le jeu d\'entrainement est de :', model_dtr.score(X_train, y_train))
            st.write('Le score sur le jeu de test est de :', model_dtr.score(X_test, y_test))
            st.write('ici nous avons 100% des données qui sont expliquées par le modèle sur le jeu d\'entrainement et 91% sur le jeu de test. Nous avons clairement à faire à un surapprentissage')
            st.subheader('Analyse des prédictions')
            score_train_dtr =model_dtr.score(X_train, y_train)
            score_test_dtr = model_dtr.score(X_test, y_test)
            
            y_pred_dtr = model_dtr.predict(X_test)
            erreurs_dtr = y_pred_dtr - y_test
            errors_dtr = pd.DataFrame({'realite' : y_test,
                                       'predictions' : y_pred_dtr,
                                       'erreurs de prediction' : erreurs_dtr})
            errors_dtr['qualité prédiction'] = pd.cut(errors_dtr['erreurs de prediction'],bins =[-0.8, -0.1, 0, 0.1, 0.8], labels = [' mauvaise (-)', 'tolérée (-)', 'tolérée (+)', 'mauvaise(+)'])
            
    # graphique des prédictions
    
            import plotly.graph_objects as go
            fig42 = go.Figure()

    
            fig42.add_trace(go.Scatter(x=data_modelisation['year'], y=target,
                                mode='markers', name='Données réelles', marker=dict(color='cornflowerblue')))
            fig42.add_trace(go.Scatter(x=data_modelisation['year'], y=model_dtr.predict(feats),
                                mode='lines', name='Prédictions', line=dict(color='red')))

            fig42.update_layout(title='Prédictions du dtr',
                        xaxis_title='Années',
                        yaxis_title='Anomalies °C',
                        showlegend=True,
                        template='plotly_white')

            st.plotly_chart(fig42)
            
            
            
            st.write('Il semblerait que le modèle réussisse à prédire une bonne partie des données.')
            
       
        
            
            st.write(' ')
            st.write('')
            st.subheader('Les résidus')
            
            
    # les metriques        
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            st.write('La MAE (Somme des valeurs absolues des écarts des erreurs) est de :', mean_absolute_error(y_test, y_pred_dtr))
            st.write('Le MSE (moyenne des distances entre les vraies valeurs et les valeurs prédites, élevées au carré) est de :', mean_squared_error(y_test, y_pred_dtr))
            st.write('La RMSE (Racine carrée de la MSE) est de :', mean_squared_error(y_test, y_pred_dtr, squared = False))
            st.write('')
            mae_dtr =  mean_absolute_error(y_test, y_pred_dtr)
            mse_dtr = mean_squared_error(y_test, y_pred_dtr)
            rmse_dtr = mean_squared_error(y_test, y_pred_dtr, squared = False)
            
   # graphique sur le resultat des predictions et ses erreurs
            
            erreurs_predictions_dtr =  model_dtr.predict(feats)-target
            erreur_pred_dtr = pd.DataFrame({'realité' : target, 
                                            'predictions' : model_dtr.predict(feats), 
                                            'erreur de prediction' : erreurs_predictions_dtr})
            erreur_pred_dtr['qualite prediction']= pd.cut(erreurs_predictions_dtr, bins =[ -0.8,-0.10,-0.02,0.02,0.10, 0.8], labels = ['mauvaise (-)', 'tolérée (-)', 'parfaite', 'tolérée (+)', 'mauvaise(+)'])
            import plotly.express as px
            
            var_x = 'realité'
            var_z = 'erreur de prediction'
            var_y = 'predictions'
            var_t = 'qualite prediction'
          
            fig50 = px.scatter(
                data_frame = erreur_pred_dtr,
                x = var_x, 
                y= var_y,
                color = var_z,
                
                title = 'Resultat des predictions et ses erreurs')
           
    # countplot sur les erreurs     
            
            st.plotly_chart(fig50, theme = None)
            
            st.write('Ici nous considérons une erreur de prédiction entre -0.02°C et 0.02°C comme parfaite, et nous tolérons les erreurs inférieures à 0,10 °C, considérant toutes les autres erreurs comme mauvaises')
            fig21 = plt.figure()
            sns.countplot(erreur_pred_dtr, x= 'qualite prediction', color = 'cornflowerblue')
            sns.color_palette('flare', as_cmap = True)
            st.pyplot(fig21)
                
            st.write('le modèle fait dans sa majorité de bonnes prédictions, mais il ne serait pas en mesure d\'être capable d\'apprendre sur de nouvelles données')
            
            
        if Modeles == 'Random Forest': 
            st.header('Le Random Forest')
                #entrainement du modele 
            feats = data_modelisation['year'].values.reshape(-1,1)
            target = data_modelisation['Anomalies °C']
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state = 0)
            from sklearn.ensemble import RandomForestRegressor
            model_rfr = RandomForestRegressor()
            model_rfr.fit(X_train, y_train)
            st.subheader('Résultats du modèle')
            st.write('Le score sur le jeu d\'entrainement est de :', model_rfr.score(X_train, y_train))
            st.write('Le score sur le jeu de test est de :', model_rfr.score(X_test, y_test))
            st.write('ici nous avons 99% des données qui sont expliquées par le modèle sur le jeu d\'entrainement et 93% sur le jeu de test. Nous avons clairement à faire à un surapprentissage')
            st.subheader('Analyse des prédictions')
            score_train_rfr = model_rfr.score(X_train, y_train)
            score_test_rfr = model_rfr.score(X_test, y_test)
                
            y_pred_rfr = model_rfr.predict(X_test)
            erreurs_rfr = y_pred_rfr - y_test
            errors_rfr = pd.DataFrame({'realite' : y_test,
                                       'predictions' : y_pred_rfr,
                                       'erreurs de prediction' : erreurs_rfr})
            errors_rfr['qualité prédiction'] = pd.cut(errors_rfr['erreurs de prediction'],bins =[-0.8, -0.1, 0, 0.1, 0.8], labels = ['mauvaise (-)', ' tolérée (-)', 'tolérée (+)', 'mauvaise (+)'])
            
    # graphique des predictions 
            
            import plotly.graph_objects as go
            fig41 = go.Figure()

    
            fig41.add_trace(go.Scatter(x=data_modelisation['year'], y=target,
                                mode='markers', name='Données réelles', marker=dict(color='blue')))

        
            fig41.add_trace(go.Scatter(x=data_modelisation['year'], y=model_rfr.predict(feats),
                                mode='lines', name='Prédictions', line=dict(color='red')))

            fig41.update_layout(title='Prédictions du rfr',
                    xaxis_title='Années',
                    yaxis_title='Anomalies °C',
                    showlegend=True,
                    template='plotly_white')

            st.plotly_chart(fig41)
            
            
            
            
            import plotly.express as px
            
            
            st.write('')
            st.subheader('Les résidus')
    # les metriques
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            
            st.write('La MAE (Somme des valeurs absolues des écarts des erreurs) est de :', mean_absolute_error(y_test, y_pred_rfr))
            st.write('Le MSE (moyenne des distances entre les vraies valeurs et les valeurs prédites, élevées au carré) est de :', mean_squared_error(y_test, y_pred_rfr))
            st.write('La RMSE (Racine carrée de la MSE) est de :', mean_squared_error(y_test, y_pred_rfr, squared = False))
            st.write('')
            mae_rfr = mean_absolute_error(y_test, y_pred_rfr)
            mse_rfr = mean_squared_error(y_test, y_pred_rfr)
            rmse_rfr= mean_squared_error(y_test, y_pred_rfr, squared = False)
            
    # graphique des erreurs 
            erreurs_predictions_rfr = model_rfr.predict(feats)- target
            erreur_pred_rfr = pd.DataFrame({'realité' : target, 
                                            'predictions' : model_rfr.predict(feats), 
                                            'erreur de prediction' : erreurs_predictions_rfr})
            erreur_pred_rfr['qualite prediction']= pd.cut(erreurs_predictions_rfr, bins =[ -0.8,-0.10,-0.02,0.02,0.10, 0.8], labels = ['mauvaise (-)', 'tolérée (-)', 'parfaite', 'tolérée (+)', 'mauvaise(+)'])
            import plotly.express as px
            
            var_x = 'realité'
            var_z = 'erreur de prediction'
            var_y = 'predictions'
            var_t = 'qualite prediction'
          
            fig51 = px.scatter(
                data_frame = erreur_pred_rfr,
                x = var_x, 
                y= var_y,
                color = var_z,
                
                title = 'Resultat des predictions et ses erreurs')
            st.plotly_chart(fig51, theme = None)
            
            
    # countplot des residus    
            st.write('Ici nous considérons une erreur de prédiction entre -0.02°C et 0.02°C comme parfaite, et nous tolérons les erreurs inférieures à 0,10 °C, considérant toutes les autres erreurs comme mauvaises')
            fig22 = plt.figure()
            sns.countplot(erreur_pred_rfr, x= 'qualite prediction', color = 'cornflowerblue')
            sns.color_palette('flare', as_cmap = True)
            st.pyplot(fig22)
                
            
            
        
            
            st.write('Aucun modèle ne serait donc concluant en l\'état afin de réaliser des prédictions sur les anomalies de températures futures. Celui qui ferait le moins d\'erreurs de prédictions serait le decision tree. Voyons comment nous pourrions contrer l\'overfitting')
            
        if Modeles == 'Lasso':
            
            st.subheader(' Analyse des résultats ')
            from sklearn.model_selection import train_test_split
            feats = data_modelisation['year'].values.reshape(-1,1)
            target = data_modelisation['Anomalies °C']
            X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state = 0)
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            from sklearn.linear_model import LassoCV, Lasso
            model_lasso = LassoCV(cv = 5, random_state = 0, max_iter = 10000)
            model_lasso.fit(X_train_scaled, y_train)
            st.write('La meilleure valeur alpha est : ', model_lasso.alpha_)
            model_lasso = Lasso(alpha = model_lasso.alpha_)
            model_lasso.fit(X_train_scaled, y_train)
            st.write('le score sur le jeu d\'entrainement est de :',  model_lasso.score(X_train_scaled, y_train))
            st.write('le score sur le jeu de test est de:', model_lasso.score(X_test_scaled, y_test))
            st.write('Ici on a 74% des données qui sont expliquées sur le jeu d\'entraînement et 71% sur le jeu de test')
            model_lasso_train_score = model_lasso.score(X_train_scaled, y_train)
            model_lasso_test_score = model_lasso.score(X_test_scaled, y_test)
            y_pred_lasso = model_lasso.predict(X_test_scaled)
            
            erreurs_lasso = y_pred_lasso - y_test
            errors_lasso = pd.DataFrame({'realite' : y_test,
                                         'predictions' : y_pred_lasso,
                                         'erreurs de prediction' : erreurs_lasso})
            errors_lasso['qualité prédiction'] = pd.cut(errors_lasso['erreurs de prediction'],bins =[-0.8, -0.1, 0, 0.1, 0.8], labels = ['mauvaise (-)', ' tolérée (-)', 'tolérée (+)', 'mauvaise (+)'])
            scaled_feats = scaler.fit_transform(feats)
            
            st.subheader('Analyse des predictions')
            import plotly.graph_objects as go
            fig42 = go.Figure()


            fig42.add_trace(go.Scatter(x=data_modelisation['year'], y=target,
                           mode='markers', name='Données réelles', marker=dict(color='cornflowerblue')))

   
            fig42.add_trace(go.Scatter(x=data_modelisation['year'], y=model_lasso.predict(scaled_feats),
                           mode='lines', name='Prédictions', line=dict(color='red')))

            fig42.update_layout(title='Prédictions Lasso',
               xaxis_title='Années',
               yaxis_title='Anomalies °C',
               showlegend=True,
               template='plotly_white')

            st.plotly_chart(fig42)
       
       
       
          
            
            import plotly.express as px
            
            
            
          
            
            st.subheader('Les résidus')
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            
            st.write('La MAE (Somme des valeurs absolues des écarts des erreurs) est de :', mean_absolute_error(y_test, y_pred_lasso))
            st.write('Le MSE (moyenne des distances entre les vraies valeurs et les valeurs prédites, élevées au carré) est de :', mean_squared_error(y_test, y_pred_lasso))
            st.write('La RMSE (Racine carrée de la MSE) est de :', mean_squared_error(y_test, y_pred_lasso, squared = False))
            st.write('')
            mae_lasso =mean_absolute_error(y_test, y_pred_lasso)
            mse_lasso = mean_squared_error(y_test, y_pred_lasso)
            rmse_lasso = mean_squared_error(y_test, y_pred_lasso, squared = False)
            
            
            
            
            
            erreurs_predictions_lasso =  model_lasso.predict(scaled_feats) - target
            erreur_pred_lasso = pd.DataFrame({'realité' : target, 
                                            'predictions' : model_lasso.predict(scaled_feats), 
                                            'erreur de prediction' : erreurs_predictions_lasso})
            erreur_pred_lasso['qualite prediction']= pd.cut(erreurs_predictions_lasso, bins =[ -0.8,-0.10,-0.02,0.02,0.10, 0.8], labels = ['mauvaise (-)', 'tolérée (-)', 'parfaite', 'tolérée (+)', 'mauvaise(+)'])
            import plotly.express as px
            
            var_x = 'realité'
            var_z = 'erreur de prediction'
            var_y = 'predictions'
            var_t = 'qualite prediction'
          
            fig52 = px.scatter(
                data_frame = erreur_pred_lasso,
                x = var_x, 
                y= var_y,
                color = var_z,
                
                title = 'Resultat des predictions et ses erreurs')
            st.plotly_chart(fig52, theme = None)
            
            
    # countplot des residus    
            st.write('Ici nous considérons une erreur de prédiction entre -0.02°C et 0.02°C comme parfaite, et nous tolérons les erreurs inférieures à 0,10 °C, considérant toutes les autres erreurs comme mauvaises')
            fig23 = plt.figure()
            sns.countplot(erreur_pred_lasso, x= 'qualite prediction', color = 'cornflowerblue')
            sns.color_palette('flare', as_cmap = True)
            st.pyplot(fig23)
   
            
        if Modeles == 'Polynomial':
            
            st.subheader('Analyse des résultats')
            from sklearn.model_selection import train_test_split
            feats = data_modelisation['year'].values.reshape(-1,1)
            target = data_modelisation['Anomalies °C']
            X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state = 0)
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.pipeline import make_pipeline
            import plotly.graph_objects as go
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            from sklearn.linear_model import LinearRegression
            degree = 3  # Degré du polynôme
            model_poly = make_pipeline(PolynomialFeatures(degree), LinearRegression())

            # Entraînement du modèle
            model_poly.fit(X_train, y_train)

            # Score sur l'ensemble d'entraînement
            st.write('le score sur le jeu d\'entrainement est de:', model_poly.score(X_train, y_train))
            st.write('le score sur le jeu de test est de :', model_poly.score(X_test, y_test))
            
            st.write('Ici, on a 87% des données qui sont expliquées par le modèle sur le jeu d\'entraînement et 86% sur le jeu de test')
            score_train_poly = model_poly.score(X_train, y_train)
            score_test_poly = model_poly.score(X_test, y_test)


            # Prédictions sur l'ensemble de test
            y_pred_poly = model_poly.predict(X_test)
            
            
            erreurs_poly = y_pred_poly - y_test
            
            errors_poly = pd.DataFrame({'realite' : y_test,
                                         'predictions' : y_pred_poly,
                                         'erreurs de prediction' : erreurs_poly})
            errors_poly['qualité prédiction'] = pd.cut(errors_poly['erreurs de prediction'],bins =[-0.8, -0.10, 0, 0.10, 0.8], labels = ['mauvaise (-)', ' tolérée (-)', 'tolérée (+)', 'mauvaise (+)'])
            
            st.subheader('Analyse des prédictions')
            
            import plotly.graph_objects as go
            fig45= go.Figure()

        # Ajouter les données réelles
            fig45.add_trace(go.Scatter(x=data_modelisation['year'], y = target,
                                mode='markers', name='Données réelles', marker=dict(color='cornflowerblue')))

        # Ajouter les prédictions du modèle
            fig45.add_trace(go.Scatter(x=data_modelisation['year'], y=model_poly.predict(feats),
                                mode='lines', name='Prédictions', line=dict(color='red')))

        # Mise en page du graphique
            fig45.update_layout(title='Prédictions de polynomial',
                        xaxis_title='Années',
                        yaxis_title='Anomalies °C',
                        showlegend=True,
                        template='plotly_white')

        # Afficher le graphique dans Streamlit
            st.plotly_chart(fig45)
            
            
            st.write('La MAE (Somme des valeurs absolues des écarts des erreurs) est de :', mean_absolute_error(y_test, y_pred_poly))
            st.write('La MSE (moyenne des distances entre les vraies valeurs et les valeurs prédites, élevées au carré) est de :', mean_squared_error(y_test, y_pred_poly))
            st.write('La RMSE (Racine carrée de la MSE) est de :', mean_squared_error(y_test, y_pred_poly, squared = False))
            st.write('')
            mae_poly =  mean_absolute_error(y_test, y_pred_poly)
            mse_poly = mean_squared_error(y_test, y_pred_poly)
            rmse_poly = mean_squared_error(y_test, y_pred_poly, squared = False)
            
            import plotly.express as px
            
            erreurs_predictions_poly =  model_poly.predict(feats)- target
            erreur_pred_poly = pd.DataFrame({'realité' : target, 
                                            'predictions' : model_poly.predict(feats), 
                                            'erreur de prediction' : erreurs_predictions_poly})
            erreur_pred_poly['qualite prediction']= pd.cut(erreurs_predictions_poly, bins =[ -0.8,-0.10,-0.02,0.02,0.10, 0.8], labels = ['mauvaise (-)', 'tolérée (-)', 'parfaite', 'tolérée (+)', 'mauvaise(+)'])
            import plotly.express as px
            
            var_x = 'realité'
            var_z = 'erreur de prediction'
            var_y = 'predictions'
            var_t = 'qualite prediction'
          
            fig53 = px.scatter(
                data_frame = erreur_pred_poly,
                x = var_x, 
                y= var_y,
                color = var_z,
                
                title = 'Resultat des predictions et ses erreurs')
            st.plotly_chart(fig53, theme = None)
            
            
    # countplot des residus    
            st.write('Ici nous considérons une erreur de prédiction entre -0.02°C et 0.02°C comme parfaite, et nous tolérons les erreurs inférieures à 0,10 °C, considérant toutes les autres erreurs comme mauvaises')
            fig24 = plt.figure()
            sns.countplot(erreur_pred_poly, x= 'qualite prediction', color = 'cornflowerblue')
            sns.color_palette('flare', as_cmap = True)
            st.pyplot(fig24)
           

          
        
            
            if st.button('En conclusion'):
                st.write('')
                st.write('')
                            
            
           
                st.subheader('4 - En conclusion')
                st.write('')
                st.write('')
                                
              
                
                st.write('Au regard des évaluations des différents modèles, le modele polynomiale serait le moins incertain des modèles de prédictions des anomalies de températures dans l\'hémisphère Nord. L\'enjeu etait ici de trouver le modèle qui produirait le moins de résidus sur les valeurs élevées au regard de la particularité de cet hémisphère.')
        
        
   
            
           
        if Modeles_CO2 == 'Regression lineaire':
            df=pd.read_csv("/Users/maison/Documents/PROJET/NH.Ts+dSST(3).csv", header = 1)
            co2 = pd.read_csv('/Users/maison/Documents/PROJET/owid-co2-data(1).csv') 
            co2 = co2[['year','country' ,'co2_including_luc']]
            co2 = co2.loc[co2['year']>= 1880]
            co2 = co2.loc[co2['country'] == 'World']
            co2 = co2.drop('country', axis = 1)
            co2 = co2.rename(columns = {'co2_including_luc' : 'CO2'})
            df = df.rename(columns = {'Year': 'year',
                                                  'J-D' : 'Anomalies °C'})
            data = pd.merge(df, co2, on = 'year')    
            data = data.drop('SON', axis = 1) 
            data = data.drop(['D-N', 'DJF', 'MAM', 'JJA'], axis = 1)
            data_modelisation = data.drop(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], axis = 1)
            data_modelisation_CO2 = data_modelisation[['year', 'CO2']]
            
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
                
            feats = data_modelisation_CO2['year'].values.reshape(-1,1)
            target = data_modelisation_CO2['CO2']
                
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state = 0)
            lr.fit(X_train, y_train)
            st.subheader('Resultats du modèle')

            st.write('le score sur le jeu d\'entraînement est de :', lr.score(X_train, y_train))
            st.write('le score sur le jeu de test est de :', lr.score(X_test, y_test))
                
                
            st.subheader('Analyse des prédictions')
        # graphique des predictions 
            y_pred  = lr.predict(X_test)
                
                
                
            import plotly.graph_objects as go
            fig43 = go.Figure()

   
            fig43.add_trace(go.Scatter(x=data_modelisation_CO2['year'], y=target,
                               mode='markers', name='Données réelles', marker=dict(color='cornflowerblue')))

       
            fig43.add_trace(go.Scatter(x=data_modelisation_CO2['year'], y= lr.predict(feats), mode='lines', name='Prédictions', line=dict(color='red')))

            fig43.update_layout(title='Prédictions regression lineaire',
                   xaxis_title='Années',
                   yaxis_title='Anomalies °C',
                   showlegend=True,
                   template='plotly_white')

            st.plotly_chart(fig43)
            
            st.subheader('Les résidus')
                
        # calcul métriques 
                
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            st.write('La MAE (Somme des valeurs absolues des écarts des erreurs) est de  :', mean_absolute_error(y_test, y_pred))
            st.write('La MSE (moyenne des distances entre les vraies valeurs et les valeurs prédites, élevées au carré) est de :', mean_squared_error(y_test, y_pred))
            st.write('La RMSE (Racine carrée de la MSE) est de :', mean_squared_error(y_test, y_pred), squared = False)
                
           
        # graphique des erreurs
        
            import plotly.express as px
                
            erreurs_predictions_lr =  lr.predict(feats)- target
            erreur_pred_lr = pd.DataFrame({'realité' : target, 
                                               'predictions' : lr.predict(feats), 
                                                'erreur de prediction' : erreurs_predictions_lr})
            erreur_pred_lr['qualite prediction']= pd.cut(erreurs_predictions_lr, bins =[ -10000000,-2000,-1000,1000,2000, 100000000], labels = ['mauvaise (-)', 'tolérée (-)', 'parfaite', 'tolérée (+)', 'mauvaise(+)'])
            import plotly.express as px
                
            var_x = 'realité'
            var_z = 'erreur de prediction'
            var_y = 'predictions'
            var_t = 'qualite prediction'
              
            fig53 = px.scatter(
                data_frame = erreur_pred_lr,
                    x = var_x, 
                    y= var_y,
                    color = var_z,
                    
                    title = 'Resultat des predictions et ses erreurs')
            st.plotly_chart(fig53, theme = None)
                
        # countplot
        
            st.write('Ici nous considérons une erreur de prédiction entre - 1000k tonnes et 1000k tonnes comme parfaite, et nous tolérons les erreurs inférieures à 2000k tonnes, considérant toutes les autres erreurs comme mauvaises')
            fig24 = plt.figure()
            sns.countplot(erreur_pred_lr, x= 'qualite prediction', color = 'cornflowerblue')
            sns.color_palette('flare', as_cmap = True)
            st.pyplot(fig24)
                
                
        if Modeles_CO2 == 'Decision tree':
            df=pd.read_csv("/Users/maison/Documents/PROJET/NH.Ts+dSST(3).csv", header = 1)
            co2 = pd.read_csv('/Users/maison/Documents/PROJET/owid-co2-data(1).csv') 
            co2 = co2[['year','country' ,'co2_including_luc']]
            co2 = co2.loc[co2['year']>= 1880]
            co2 = co2.loc[co2['country'] == 'World']
            co2 = co2.drop('country', axis = 1)
            co2 = co2.rename(columns = {'co2_including_luc' : 'CO2'})
            df = df.rename(columns = {'Year': 'year',
                                                  'J-D' : 'Anomalies °C'})
            data = pd.merge(df, co2, on = 'year')    
            data = data.drop('SON', axis = 1) 
            data = data.drop(['D-N', 'DJF', 'MAM', 'JJA'], axis = 1)
            data_modelisation = data.drop(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], axis = 1)
            data_modelisation_CO2 = data_modelisation[['year', 'CO2']]
            from sklearn.tree import DecisionTreeRegressor
            dtr = DecisionTreeRegressor()
             
            feats = data_modelisation_CO2['year'].values.reshape(-1,1)
            target = data_modelisation_CO2['CO2']
             
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state = 0)
            dtr.fit(X_train, y_train)
            st.subheader('Analyse des résultats')

            st.write('le score sur le jeu d\'entraînement est de :', dtr.score(X_train, y_train))
            st.write('le score sur le jeu de test est de :', dtr.score(X_test, y_test))
             
            st.subheader('Analyse des prédictions')
     # graphique des predictions 
            y_pred  = dtr.predict(X_test)
             
             
             
            import plotly.graph_objects as go
            fig43 = go.Figure()


            fig43.add_trace(go.Scatter(x=data_modelisation_CO2['year'], y=target,
                            mode='markers', name='Données réelles', marker=dict(color='cornflowerblue')))

    
            fig43.add_trace(go.Scatter(x=data_modelisation_CO2['year'], y= dtr.predict(feats), mode='lines', name='Prédictions', line=dict(color='red')))

            fig43.update_layout(title='Prédictions dtr pour CO2',
                                     xaxis_title='Années',
                                     yaxis_title='Anomalies °C',
                                     showlegend=True,
                                     template='plotly_white')

            st.plotly_chart(fig43)
            st.subheader('Les résidus')
     # calcul métriques 
             
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            st.write('La MAE (Somme des valeurs absolues des écarts des erreurs) est de :', mean_absolute_error(y_test, y_pred))
            st.write('La MSE (moyenne des distances entre les vraies valeurs et les valeurs prédites, élevées au carré) est de :', mean_squared_error(y_test, y_pred))
            st.write('La RMSE (Racine carrée de la MSE) est de :', mean_squared_error(y_test, y_pred), squared = False)
             
        
     # graphique des erreurs
     
            import plotly.express as px
             
            erreurs_predictions_dtr=  dtr.predict(feats)- target
            erreur_pred_dtr = pd.DataFrame({'realité' : target, 
                                             'predictions' : dtr.predict(feats), 
                                             'erreur de prediction' : erreurs_predictions_dtr})
            erreur_pred_dtr['qualite prediction']= pd.cut(erreurs_predictions_dtr, bins =[ -10000000,-2000,-1000,1000,2000, 100000000], labels = ['mauvaise (-)', 'tolérée (-)', 'parfaite', 'tolérée (+)', 'mauvaise(+)'])
            import plotly.express as px
             
            var_x = 'realité'
            var_z = 'erreur de prediction'
            var_y = 'predictions'
            var_t = 'qualite prediction'
           
            fig53 = px.scatter(
                     data_frame = erreur_pred_dtr,
                     x = var_x, 
                     y= var_y,
                     color = var_z,
                 
                    title = 'Resultat des predictions et ses erreurs')
            st.plotly_chart(fig53, theme = None)
             
     # countplot
     
            st.write('Ici nous considérons une erreur de prédiction entre - 1000k tonnes et 1000k tonnes comme parfaite, et nous tolérons les erreurs inférieures à 2000k tonnes, considérant toutes les autres erreurs comme mauvaises')
            fig24 = plt.figure()
            sns.countplot(erreur_pred_dtr, x= 'qualite prediction', color = 'cornflowerblue')
            sns.color_palette('flare', as_cmap = True)
            st.pyplot(fig24)
                
        if Modeles_CO2 =='Random Forest':
            df=pd.read_csv("/Users/maison/Documents/PROJET/NH.Ts+dSST(3).csv", header = 1)
            co2 = pd.read_csv('/Users/maison/Documents/PROJET/owid-co2-data(1).csv') 
            co2 = co2[['year','country' ,'co2_including_luc']]
            co2 = co2.loc[co2['year']>= 1880]
            co2 = co2.loc[co2['country'] == 'World']
            co2 = co2.drop('country', axis = 1)
            co2 = co2.rename(columns = {'co2_including_luc' : 'CO2'})
            df = df.rename(columns = {'Year': 'year',
                                                  'J-D' : 'Anomalies °C'})
            data = pd.merge(df, co2, on = 'year')    
            data = data.drop('SON', axis = 1) 
            data = data.drop(['D-N', 'DJF', 'MAM', 'JJA'], axis = 1)
            data_modelisation = data.drop(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], axis = 1)
            data_modelisation_CO2 = data_modelisation[['year', 'CO2']]
                

            from sklearn.ensemble import RandomForestRegressor
            rfr =RandomForestRegressor()
            
                    
            feats = data_modelisation_CO2['year'].values.reshape(-1,1)
            target = data_modelisation_CO2['CO2']
            st.subheader('Résultats du modèle')
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state = 0)
            rfr.fit(X_train, y_train)
            st.write('le score sur le jeu d\'entraînement est de :', rfr.score(X_train, y_train))
            st.write('le score sur le jeu de test est de :', rfr.score(X_test, y_test))
                    
            st.subheader('Analyse des prédictions')     
            # graphique des predictions 
            y_pred  = rfr.predict(X_test)
                    
                    
                    
            import plotly.graph_objects as go
            fig43 = go.Figure()

       
            fig43.add_trace(go.Scatter(x=data_modelisation_CO2['year'], y=target,
                                   mode='markers', name='Données réelles', marker=dict(color='cornflowerblue')))

           
            fig43.add_trace(go.Scatter(x=data_modelisation_CO2['year'], y= rfr.predict(feats), mode='lines', name='Prédictions', line=dict(color='red')))

            fig43.update_layout(title='Prédictions rfr pour CO2',
                       xaxis_title='Années',
                       yaxis_title='Anomalies °C',
                       showlegend=True,
                       template='plotly_white')

            st.plotly_chart(fig43)
            st.subheader('Les résidus')
            # calcul métriques 
                    
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            st.write('La MAE (Somme des valeurs absolues des écarts des erreurs) est de :', mean_absolute_error(y_test, y_pred))
            st.write('La MSE (moyenne des distances entre les vraies valeurs et les valeurs prédites, élevées au carré) est de :', mean_squared_error(y_test, y_pred))
            st.write('La RMSE (Racine carrée de la MSE) est de :', mean_squared_error(y_test, y_pred), squared = False)
                    
               
            # graphique des erreurs
            
            import plotly.express as px
                    
            erreurs_predictions_rfr =  rfr.predict(feats)- target
            erreur_pred_rfr = pd.DataFrame({'realité' : target, 
                                                    'predictions' : rfr.predict(feats), 
                                                    'erreur de prediction' : erreurs_predictions_rfr})
            erreur_pred_rfr['qualite prediction']= pd.cut(erreurs_predictions_rfr, bins =[ -10000000,-2000,-1000,1000,2000, 100000000], labels = ['mauvaise (-)', 'tolérée (-)', 'parfaite', 'tolérée (+)', 'mauvaise(+)'])
            import plotly.express as px
                    
            var_x = 'realité'
            var_z = 'erreur de prediction'
            var_y = 'predictions'
            var_t = 'qualite prediction'
                  
            fig53 = px.scatter(
                data_frame = erreur_pred_rfr,
                x = var_x, 
                y= var_y,
                color = var_z,
                        
                title = 'Resultat des predictions et ses erreurs')
            st.plotly_chart(fig53, theme = None)
                    
            # countplot
            
            st.write('Ici nous considérons une erreur de prédiction entre - 1000k tonnes et 1000k tonnes comme parfaite, et nous tolérons les erreurs inférieures à 2000k tonnes, considérant toutes les autres erreurs comme mauvaises')
            fig24 = plt.figure()
            sns.countplot(erreur_pred_rfr, x= 'qualite prediction', color = 'cornflowerblue')
            sns.color_palette('flare', as_cmap = True)
            st.pyplot(fig24)
            
            
        if Modeles_CO2 == 'polynomial':
            df=pd.read_csv("/Users/maison/Documents/PROJET/NH.Ts+dSST(3).csv", header = 1)
            co2 = pd.read_csv('/Users/maison/Documents/PROJET/owid-co2-data(1).csv') 
            co2 = co2[['year','country' ,'co2_including_luc']]
            co2 = co2.loc[co2['year']>= 1880]
            co2 = co2.loc[co2['country'] == 'World']
            co2 = co2.drop('country', axis = 1)
            co2 = co2.rename(columns = {'co2_including_luc' : 'CO2'})
            df = df.rename(columns = {'Year': 'year',
                                                  'J-D' : 'Anomalies °C'})
            data = pd.merge(df, co2, on = 'year')    
            data = data.drop('SON', axis = 1) 
            data = data.drop(['D-N', 'DJF', 'MAM', 'JJA'], axis = 1)
            data_modelisation = data.drop(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], axis = 1)
            data_modelisation_CO2 = data_modelisation[['year', 'CO2']]
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
                
            feats = data_modelisation_CO2['year'].values.reshape(-1,1)
            target = data_modelisation_CO2['CO2']
            X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state = 0)
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.pipeline import make_pipeline
            import plotly.graph_objects as go
            from sklearn.metrics import mean_squared_error
                
            degree = 3  # Degré du polynôme
            model_poly = make_pipeline(PolynomialFeatures(degree), LinearRegression())

                # Entraînement du modèle
            model_poly.fit(X_train, y_train)
            st.subheader('Les resultats du modèle')

                # Score sur l'ensemble d'entraînement
            st.write('le score sur le jeu d\'entrainement est de:', model_poly.score(X_train, y_train))
            st.write('le score sur le jeu de test est de :', model_poly.score(X_test, y_test))
              

                # Prédictions sur l'ensemble de test
            y_pred_poly = model_poly.predict(X_test)
                
               
                
            st.subheader('Analyse des prédictions')   
        # graphique des predictions 
            y_pred  = model_poly.predict(X_test)
                
                
                
            import plotly.graph_objects as go
            fig43 = go.Figure()

   
            fig43.add_trace(go.Scatter(x=data_modelisation_CO2['year'], y=target,
                               mode='markers', name='Données réelles', marker=dict(color='cornflowerblue')))

       
            fig43.add_trace(go.Scatter(x=data_modelisation_CO2['year'], y= model_poly.predict(feats), mode='lines', name='Prédictions', line=dict(color='red')))

            fig43.update_layout(title='Prédictions polyomial Model pour CO2',
                   xaxis_title='Années',
                   yaxis_title='Anomalies °C',
                   showlegend=True,
                   template='plotly_white')

            st.plotly_chart(fig43)
            st.subheader('Les résidus')
        # calcul métriques 
                
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            st.write('La MAE (Somme des valeurs absolues des écarts des erreurs) est de :', mean_absolute_error(y_test, y_pred))
            st.write('La MSE (moyenne des distances entre les vraies valeurs et les valeurs prédites, élevées au carré) est de :', mean_squared_error(y_test, y_pred))
            st.write('La RMSE (Racine carrée de la MSE) est de :', mean_squared_error(y_test, y_pred), squared = False)
                
           
        # graphique des erreurs
        
            import plotly.express as px
                
            erreurs_predictions_poly =  model_poly.predict(feats)- target
            erreur_pred_poly = pd.DataFrame({'realité' : target, 
                                                'predictions' : model_poly.predict(feats), 
                                                'erreur de prediction' : erreurs_predictions_poly})
            erreur_pred_poly['qualite prediction']= pd.cut(erreurs_predictions_poly, bins =[ -10000000,-2000,-1000,1000,2000, 100000000], labels = ['mauvaise (-)', 'tolérée (-)', 'parfaite', 'tolérée (+)', 'mauvaise(+)'])
            import plotly.express as px
                
            var_x = 'realité'
            var_z = 'erreur de prediction'
            var_y = 'predictions'
            var_t = 'qualite prediction'
              
            fig53 = px.scatter(
                    data_frame = erreur_pred_poly,
                    x = var_x, 
                    y= var_y,
                    color = var_z,
                    
                    title = 'Resultat des predictions et ses erreurs')
            st.plotly_chart(fig53, theme = None)
                
        # countplot
        
            st.write('Ici nous considérons une erreur de prédiction entre - 1000k tonnes et 1000k tonnes comme parfaite, et nous tolérons les erreurs inférieures à 2000k tonnes, considérant toutes les autres erreurs comme mauvaises')
            fig24 = plt.figure()
            sns.countplot(erreur_pred_poly, x= 'qualite prediction', color = 'cornflowerblue')
            sns.color_palette('flare', as_cmap = True)
            st.pyplot(fig24)
        
                
            
            
            
# PREDICTIONS
            st.write('C\'est donc le modèle polynomial qui est le plus adapté aux prédictions de CO2.')
            
            
            
    if st.checkbox('PREDICTIONS'):
        
       
           
        
        st.write('Testons quelques prédictions à l\'aide de la régression polynomiale:')
        
        df=pd.read_csv("/Users/maison/Documents/PROJET/NH.Ts+dSST(3).csv", header = 1)
        co2 = pd.read_csv('/Users/maison/Documents/PROJET/owid-co2-data(1).csv') 
        co2 = co2[['year','country' ,'co2_including_luc']]
        co2 = co2.loc[co2['year']>= 1880]
        co2 = co2.loc[co2['country'] == 'World']
        co2 = co2.drop('country', axis = 1)
        co2 = co2.rename(columns = {'co2_including_luc' : 'CO2'})
        df = df.rename(columns = {'Year': 'year',
                                              'J-D' : 'Anomalies °C'})
        data = pd.merge(df, co2, on = 'year')    
        data = data.drop('SON', axis = 1) 
        data = data.drop(['D-N', 'DJF', 'MAM', 'JJA'], axis = 1)
        data_modelisation = data.drop(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], axis = 1)
        data_modelisation_CO2 = data_modelisation[['year', 'CO2']]
                    
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
            
        feats = data_modelisation_CO2['year'].values.reshape(-1,1)
        target = data_modelisation_CO2['CO2']
        X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state = 0)
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import make_pipeline
        import plotly.graph_objects as go
        from sklearn.metrics import mean_squared_error
        
            
        
        X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2, random_state=0)
        poly_features = PolynomialFeatures(degree=2)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
            
      
        
        Année = st.selectbox('Choisissez votre année :', (2030,2035, 2040,2045,2050))
        if Année == 2030:
           
            feats = data_modelisation['year'].values.reshape(-1, 1)
            target = data_modelisation['Anomalies °C']
            X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state = 0)
            poly_features = PolynomialFeatures(degree = 2)
            X_train_poly = poly_features.fit_transform(X_train)
            feats_poly = poly_features.transform(feats)
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            predictions_poly = model.predict(feats_poly)
            
            pred = model.predict(poly_features.transform([[2030 ]]))
            pred_round = np.round(pred, 2)
            st.write(f'Les anomalies de tempétatures de l\'hémisphère Nord seront de: {pred_round} °C en {Année}')  
                        
            
            #annees_a_predire = np.array([[2024], [2025],[2026],[2027],[2028],[2029],[2030]])
            #annees_a_predire_poly = poly_features.transform(annees_a_predire)
            #predictions_celcius = model.predict(annees_a_predire_poly)
            
            #for year, prediction in zip(annees_a_predire.flatten(), predictions_celcius):
             #       st.write(f"       - Pour l\'année {year}, la prédiction des anomalies de °C est de: {prediction:.2f}")
                
              #      st.write(f'Polynomial Regression result : {model.predict(poly_features.transform([[2025]]))}')            
        if Année == 2035:
           
            feats = data_modelisation['year'].values.reshape(-1, 1)
            target = data_modelisation['Anomalies °C']
            X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state = 0)
            poly_features = PolynomialFeatures(degree = 2)
            X_train_poly = poly_features.fit_transform(X_train)
            feats_poly = poly_features.transform(feats)
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            predictions_poly = model.predict(feats_poly)
            
            pred = model.predict(poly_features.transform([[2035]]))
            pred_round = np.round(pred, 2)
            st.write(f'Les anomalies de tempétatures de l\'hémisphère Nord seront de: {pred_round} °C en {Année}')  
                        
        if Année == 2040:
            feats = data_modelisation['year'].values.reshape(-1, 1)
            target = data_modelisation['Anomalies °C']
            X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state = 0)
            poly_features = PolynomialFeatures(degree = 2)
            X_train_poly = poly_features.fit_transform(X_train)
            feats_poly = poly_features.transform(feats)
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            predictions_poly = model.predict(feats_poly)
            pred = model.predict(poly_features.transform([[2040]]))
            pred_round = np.round(pred, 2)
            st.write(f'Les anomalies de tempétatures de l\'hémisphère Nord seront de: {pred_round} °C en {Année}')  
                        
        if Année == 2045:
            eats = data_modelisation['year'].values.reshape(-1, 1)
            target = data_modelisation['Anomalies °C']
            X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state = 0)
            poly_features = PolynomialFeatures(degree = 2)
            X_train_poly = poly_features.fit_transform(X_train)
            feats_poly = poly_features.transform(feats)
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            predictions_poly = model.predict(feats_poly)
            pred = model.predict(poly_features.transform([[2045]]))
            pred_round = np.round(pred, 2)
            st.write(f'Les anomalies de tempétatures de l\'hémisphère Nord seront de: {pred_round} °C en {Année}')  
            
        if Année == 2050:
            eats = data_modelisation['year'].values.reshape(-1, 1)
            target = data_modelisation['Anomalies °C']
            X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state = 0)
            poly_features = PolynomialFeatures(degree = 2)
            X_train_poly = poly_features.fit_transform(X_train)
            feats_poly = poly_features.transform(feats)
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            predictions_poly = model.predict(feats_poly)
            pred = model.predict(poly_features.transform([[2050]]))
            pred_round = np.round(pred, 2)
            st.write(f'Les anomalies de tempétatures de l\'hémisphère Nord seront de: {pred_round} °C en {Année}')  
            
            
        st.write('prédictions des emissions de CO2 ')    
        Année_CO2 = st.selectbox('Choisissez votre année :', (2030,2035, 2040,2045,2050), key = co2)
        
        if Année_CO2 == 2030:
            feats = data_modelisation['year'].values.reshape(-1, 1)
            target = data_modelisation['CO2']
            X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state = 0)
            poly_features = PolynomialFeatures(degree = 2)
            X_train_poly = poly_features.fit_transform(X_train)
            feats_poly = poly_features.transform(feats)
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            predictions_poly = model.predict(feats_poly)
            pred = model.predict(poly_features.transform([[2030]]))
            round_pred = np.round(pred, 2)
            st.write(f'Les emissions de CO2 seront de: {round_pred} millions tonnes en {Année_CO2}')             
        if Année_CO2 == 2035:
            feats = data_modelisation['year'].values.reshape(-1, 1)
            target = data_modelisation['CO2']
            X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state = 0)
            poly_features = PolynomialFeatures(degree = 2)
            X_train_poly = poly_features.fit_transform(X_train)
            feats_poly = poly_features.transform(feats)
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            predictions_poly = model.predict(feats_poly)
            pred = model.predict(poly_features.transform([[2035]]))
            round_pred = np.round(pred, 2)
            st.write(f'Les emissions de CO2 seront de: {round_pred} millions tonnes en {Année_CO2}')  
            
        if Année_CO2 == 2040:
            feats = data_modelisation['year'].values.reshape(-1, 1)
            target = data_modelisation['CO2']
            X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state = 0)
            poly_features = PolynomialFeatures(degree = 2)
            X_train_poly = poly_features.fit_transform(X_train)
            feats_poly = poly_features.transform(feats)
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            predictions_poly = model.predict(feats_poly)
            pred = model.predict(poly_features.transform([[2040]]))
            round_pred = np.round(pred, 2)
            st.write(f'Les emissions de CO2 seront de: {round_pred} millions tonnes en {Année_CO2}') 
            
        if Année_CO2 == 2045:
            feats = data_modelisation['year'].values.reshape(-1, 1)
            target = data_modelisation['CO2']
            X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state = 0)
            poly_features = PolynomialFeatures(degree = 2)
            X_train_poly = poly_features.fit_transform(X_train)
            feats_poly = poly_features.transform(feats)
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            predictions_poly = model.predict(feats_poly)
            pred = model.predict(poly_features.transform([[2045]]))
            round_pred = np.round(pred, 2)
            st.write(f'Les emissions de CO2 seront de: {round_pred} millions tonnes en {Année_CO2}')  
            
            
        if Année_CO2 == 2050:
            feats = data_modelisation['year'].values.reshape(-1, 1)
            target = data_modelisation['CO2']
            X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state = 0)
            poly_features = PolynomialFeatures(degree = 2)
            X_train_poly = poly_features.fit_transform(X_train)
            feats_poly = poly_features.transform(feats)
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            predictions_poly = model.predict(feats_poly)
            pred = model.predict(poly_features.transform([[2050]]))
            round_pred = np.round(pred, 2)
            
            st.write(f'Les emissions de CO2 seront de: {round_pred} millions tonnes en {Année_CO2}')  
            
           
           
       


                
if page == pages[3]:
    st.header('DANS L\'HÉMISPHÈRE SUD')
   
    
    
    import streamlit as st 
    import plotly.express as px
    import plotly.figure_factory as ff
    import pandas as pd 
    import numpy as np 
    import time
    from scipy.stats import pearsonr
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import plotly.graph_objects as go
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    def get_data():
    # Charger les données
        Temperatures = pd.read_csv("/Users/maison/Documents/PROJET/temp_hemis_sud.csv")

    # Convertir la colonne en float en remplaçant les valeurs '***' par 0 pour minimiser les biais
    
        return Temperatures
    
    def get_data_1():
        CO2 = pd.read_csv(r'/Users/maison/Documents/PROJET/carbonne.csv')
        return CO2

    def get_data_2():

        Temperatures = pd.read_csv("/Users/maison/Documents/PROJET/temp_hemis_sud.csv")
 
        Temperatures['D-N'] = pd.to_numeric(Temperatures['D-N'], errors='coerce')
        Temperatures['DJF'] = pd.to_numeric(Temperatures['DJF'], errors='coerce')

        Temperatures['D-N'].fillna(0, inplace=True)
        Temperatures['DJF'].fillna(0, inplace=True)

    # Renommer les colonnes
        Temperatures = Temperatures.rename(columns={'Year': 'year',  # Renommer la colonne 'Year' en 'year'
                                                    'J-D': 'moyenne_annuelle',
                                                    'DJF': 'moyenne_hiver',
                                                    'MAM': 'moyenne_printemps',
                                                    'JJA': 'moyenne_ete',
                                                    'SON': 'moyenne_automne'})

        # Supprimer la colonne 'D-F' du DataFrame Temp_hemis_sud
        Temperatures = Temperatures.drop('D-N', axis=1)


        CO2 = pd.read_csv(r"/Users/maison/Documents/PROJET/carbonne.csv")
        # On ne conserve les valeurs communes qu'au deux data sets
        CO2 = CO2.loc[CO2['year'] >= 1880]
        # On ne conserve que les données mondiales :
        CO2 = CO2.loc[CO2['country']=='World']
            # On isole la variable co2 et année pour faire une fusion sur l'annee:
        CO2_unique = CO2[['co2_including_luc','population','year']]
                # On renomme la colonne année de Temp_hemis_sud pour permettre la fusion sur cette colonne
        Fusion = pd.merge(Temperatures, CO2_unique, on='year')
        Fusion = Fusion.rename(columns={'co2_including_luc': 'CO2'})

        return Fusion
    image = '/Users/maison/Desktop/climatique.png'
    st.image(image)
    
    if st.checkbox('EXPLORATION'):
        Temperatures = get_data()

        st.title('Températures')
        st.write(Temperatures)

        CO2 = get_data_1()
        st.title('CO2')
        st.write(CO2)

        Fusion = get_data_2()
        st.title('Données Néttoyés')
        st.write(Fusion)
        
    if st.checkbox('DATA VISUALISATION'):
        Fusion=get_data_2()

    # Afficher le titre
        st.title('Graphiques Températures')

    # Sélection de la colonne pour le graphique
        selected_column_temp = st.selectbox('Sélectionner une colonne pour afficher le graphique Temperature', Fusion.columns.to_list())

    # Créer le boxplot avec Plotly Express
        fig_boxplot_temp = px.box(Fusion, y=selected_column_temp, title=f'Boxplot de {selected_column_temp}')

    # Créer le nuage de points avec Plotly Express
        fig_scatter_temp = px.scatter(Fusion, x='year', y=selected_column_temp,color=selected_column_temp, title=f'Nuage de points de {selected_column_temp}')

    # Créer la courbe avec Plotly Express
        fig_line_temp = px.line(Fusion, x='year', y=selected_column_temp, title=f'Courbe de {selected_column_temp}')

    # Créer le graphique en barres avec Plotly Express
        fig_barplot_temp = px.bar(Fusion, x='year', y=selected_column_temp,color=selected_column_temp, title=f'Graphique en barres de {selected_column_temp}')

    # Afficher les graphiques en fonction de la sélection de l'utilisateur
        selected_chart_temp = st.selectbox('Sélectionner un type de graphique de Températures', ['Boxplot', 'Nuage de points', 'Courbe', 'Graphique en barres'])

        if selected_chart_temp == 'Boxplot':
            st.plotly_chart(fig_boxplot_temp)
        elif selected_chart_temp == 'Nuage de points':
            st.plotly_chart(fig_scatter_temp)
        elif selected_chart_temp == 'Courbe':
            st.plotly_chart(fig_line_temp)
        elif selected_chart_temp == 'Graphique en barres':
            st.plotly_chart(fig_barplot_temp)

    #----------------------------ECART TEMPERATURE---------------------------------------------------------------------------------------------------"


        minimum_annuelle = Fusion['moyenne_annuelle'].min() #1909
        maximum_annuelle = Fusion['moyenne_annuelle'].max() #2023
        ecart_annuelle = maximum_annuelle - minimum_annuelle

        minimum_hiver = Fusion['moyenne_hiver'].min() #1917
        maximum_hiver = Fusion['moyenne_hiver'].max() #2016
        ecart_hivernale = maximum_hiver - minimum_hiver

        minimum_ete = Fusion['moyenne_ete'].min() #1911
        maximum_ete = Fusion['moyenne_ete'].max() #2023
        ecart_estivale = maximum_ete - minimum_ete

    # Définition des données
        data = {
            'Température minimale': [minimum_annuelle, minimum_hiver, minimum_ete],
            'Température maximale': [maximum_annuelle, maximum_hiver, maximum_ete],
            'Écart de température maximal': [ecart_annuelle, ecart_hivernale, ecart_estivale],
            'Année Minimum': ['1909','1917','1911'],
            'Année Maximum': ['2023','2016','2023']



        }

    # Création du DataFrame
        ecart = pd.DataFrame(data, index=['Année', 'Hiver', 'Été'])

    # Affichage du DataFrame dans Streamlit
        st.title('Ecart de Températures')
        st.write(ecart)



    #------------------------------------------------CO2------------------------------------------------------------------------------------#

        Fusion=get_data_2()

        st.title('Graphiques Co2') 
    # Sélection de la colonne pour le graphique
        selected_column_CO2 = st.selectbox('Sélectionner une colonne pour afficher le graphique Co2', Fusion.columns.to_list())

    # Créer le boxplot avec Plotly Express
        fig_boxplot_CO2 = px.box(Fusion, y=selected_column_CO2, title=f'Boxplot de {selected_column_CO2}')

    # Créer le nuage de points avec Plotly Express
        fig_scatter_CO2 = px.scatter(Fusion, x='year', y=selected_column_CO2, color=selected_column_CO2, title=f'Nuage de points de {selected_column_CO2}')

    # Créer la courbe avec Plotly Express
        fig_line_CO2 = px.line(Fusion, x='year', y=selected_column_CO2, title=f'Courbe de {selected_column_CO2}')

    # Créer le graphique en barres avec Plotly Express
        fig_barplot_CO2 = px.bar(Fusion, x='year', y=selected_column_CO2,color=selected_column_CO2, title=f'Graphique en barres de {selected_column_CO2}')

    # Afficher les graphiques en fonction de la sélection de l'utilisateur
        selected_chart_CO2 = st.selectbox('Sélectionner un type de graphique Co2', ['Boxplot', 'Nuage de points', 'Courbe', 'Graphique en barres'])

        if selected_chart_CO2 == 'Boxplot':
            st.plotly_chart(fig_boxplot_CO2)
        elif selected_chart_CO2 == 'Nuage de points':
            st.plotly_chart(fig_scatter_CO2)
        elif selected_chart_CO2 == 'Courbe':
            st.plotly_chart(fig_line_CO2)
        elif selected_chart_CO2 == 'Graphique en barres':
            st.plotly_chart(fig_barplot_CO2)

    #-------------------------------------- GRAPHIQUES CORRELATIONS-------------------------------------------------------------------------------------"



    # Charger les données initiales
        Fusion = get_data_2()

    # Liste des variables pour les corrélations
        variables = Fusion.columns.tolist()

    # Afficher le titre de l'application
        st.title(' Graphiques Corrélations')

    # Sélection des variables pour la corrélation
        variable_x = st.selectbox('Choisir la variable x', variables)
        variable_y = st.selectbox('Choisir la variable y', variables)


    # Créer la figure Plotly
        fig_CO2_2 = px.scatter(Fusion, x=variable_x, y=variable_y, hover_name='CO2', trendline='ols', size='CO2', color='CO2',
                        title=f'Relations {variable_x} et {variable_y}')

    # Mettre à jour les informations de la mise en page
        fig_CO2_2.update_layout(xaxis_title=variable_x, yaxis_title=variable_y)
        fig_CO2_2.layout.coloraxis.colorbar.title = 'Tonne de Co2'
        fig_CO2_2.update_layout(title={'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})

    # Afficher la figure Plotly dans Streamlit
        st.plotly_chart(fig_CO2_2)

    #-----------------------CORRELATIONS TEMPERATURES ET EMISSIONS DE CO2------------------------------------------------------"

        Fusion=get_data_2()

        st.title('Corrélations saisonnières')

        st.write('la p-valeur est de : 4.492770875663679e-60  et le coefficient de correlation est de : 0.9222807584860793 pour : année')
        st.write('la p-valeur est de : 4.911221874237122e-42  et le coefficient de correlation est de : 0.8549845465714758 pour : hiver')
        st.write('la p-valeur est de : 8.547882107888152e-62  et le coefficient de correlation est de : 0.9266912711813124 pour : été')

        Fusion = get_data_2()

    # Afficher le titre de l'application
        st.title('Corrélations saisonnières par périodes')
    # Définir les années de coupure pour les trois périodes
        annee_1 = 1940
        annee_2 = 1980
    # Diviser les données en trois DataFrame pour chaque période
        fusion_1 = Fusion[Fusion['year'] < annee_1]
        fusion_2 = Fusion[(Fusion['year'] >= annee_1) & (Fusion['year'] < annee_2)]
        fusion_3 = Fusion[Fusion['year'] >= annee_2]
    # Afficher les données pour chaque période
    #st.write('Données pour la première période (avant 1940) :')
    #st.write(fusion_1)
    #st.write('Données pour la deuxième période (1940 - 1980) :')
    #st.write(fusion_2)
    #st.write('Données pour la troisième période (après 1980) :')
    #st.write(fusion_3)

        liste_fusions = ['fusion_1', 'fusion_2', 'fusion_3']
        annee_corr = []
        hiver_corr = []
        ete_corr = []
        p_value_annee = []
        p_value_hiver = []
        p_value_ete = []

    # Calcul des coefficients de corrélation et des p-valeurs pour chaque fusion
        for fusion in liste_fusions:
            coef_annee, p_val_annee = pearsonr(eval(fusion)['CO2'], eval(fusion)['moyenne_annuelle'])
            coef_hiver, p_val_hiver = pearsonr(eval(fusion)['CO2'], eval(fusion)['moyenne_hiver'])
            coef_ete, p_val_ete = pearsonr(eval(fusion)['CO2'], eval(fusion)['moyenne_ete'])
        
            annee_corr.append(coef_annee)
            hiver_corr.append(coef_hiver)
            ete_corr.append(coef_ete)
            p_value_annee.append(p_val_annee)
            p_value_hiver.append(p_val_hiver)
            p_value_ete.append(p_val_ete)

        data = {
            'Année': annee_corr,
            'Hiver': hiver_corr,
            'Été': ete_corr,
            'P-value Année': p_value_annee,
            'P-value Hiver': p_value_hiver,
            'P-value Été': p_value_ete
    }

        Coeff = pd.DataFrame(data, index=liste_fusions)
        Coeff

        st.write('P-value Année pour fusion_3 : 2.895104e-13')
        st.write('P-value Hiver pour fusion_3 : 0.000019')
        st.write('P-value Eté pour fusion_3 : 3.724929e-10')


        
    if st.checkbox('MODELISATION'):
        Fusion = get_data_2()

    # Afficher le titre
        st.title('Résultats des Modèles de Prédictions')

    # Sélection du modèle
        selected_model = st.selectbox('Sélectionner le modèle', ['Régression linéaire', 'Arbre de décision', 'Random Forest'])

    # Sélection de la variable
        selected_variable = st.selectbox('Sélectionner la variable à étudier', Fusion.columns)

    # Diviser les données en X (caractéristiques) et y (cible)
        X = Fusion[['year']]
        y = Fusion[selected_variable]

    # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choisir le modèle en fonction de la sélection de l'utilisateur
        if selected_model == 'Régression linéaire':
            model = LinearRegression()
        elif selected_model == 'Arbre de décision':
            model = DecisionTreeRegressor()
        elif selected_model == 'Random Forest':
            model = RandomForestRegressor()

    # Entraîner le modèle
        model.fit(X_train, y_train)

    # Faire des prédictions
        predictions = model.predict(X_test)

    # Calculer les métriques
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

    # Afficher les métriques
        st.write('Variable étudiée :', selected_variable)
        st.write('Modèle sélectionné :', selected_model)
        st.write('Mean Squared Error:', mse)
        st.write('R²:', r2)
        st.write('Score sur ensemble train :', train_score)
        st.write('Score sur ensemble test :', test_score)

    # Afficher le graphique
        fig = go.Figure()

    # Ajouter les données réelles
        fig.add_trace(go.Scatter(x=Fusion['year'], y=y,
                            mode='markers', name='Données réelles', marker=dict(color='blue')))

    # Ajouter les prédictions du modèle
        fig.add_trace(go.Scatter(x=Fusion['year'], y=model.predict(X),
                            mode='lines', name='Prédictions', line=dict(color='red')))

    # Mise en page du graphique
        fig.update_layout(title='Prédictions de ' + selected_variable + ' en fonction du temps',
                    xaxis_title='Temps',
                    yaxis_title='Anomalies de ' + selected_variable + ' annuelles',
                    showlegend=True,
                    template='plotly_white')

    # Afficher le graphique dans Streamlit
        st.plotly_chart(fig)

        

    #------------------------------FUTUR------------------------------------------------------------"

        Fusion = get_data_2()

    # Titre
        st.title('Régression polynomiale')

    # Sélectionner la variable à étudier
        selected_variable = st.selectbox('Sélectionner la variable futur à étudier', Fusion.columns)

    # Données d'entraînement
        X = Fusion['year'].values.reshape(-1, 1)
        y = Fusion[selected_variable]

    # Transformation polynomiale des features
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

    # Entraînement du modèle de régression polynomiale
        model_poly = LinearRegression()
        model_poly.fit(X_poly, y)

    # Années futures pour les prédictions
        annees_futures = np.array([2025, 2030, 2035, 2040, 2045, 2050]).reshape(-1, 1)

    # Transformation polynomiale des années futures
        annees_futures_poly = poly.transform(annees_futures)

    # Prédictions des températures pour les années futures
        predictions_poly = model_poly.predict(annees_futures_poly)

    # Création d'un DataFrame pour les prédictions futures
        df_predictions_poly = pd.DataFrame({'year': annees_futures.flatten(), 'moyenne_annuelle_pred_poly': predictions_poly})


    # Création du modèle de régression polynomiale
        degree = 2  # Degré du polynôme
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    # Entraînement du modèle
        model.fit(X_train, y_train)

    # Score sur l'ensemble d'entraînement
        train_score = model.score(X_train, y_train)

    # Score sur l'ensemble de test
        test_score = model.score(X_test, y_test)

    # Prédictions sur l'ensemble de test
        y_pred = model.predict(X_test)

    # Calcul du Mean Squared Error
        mse = mean_squared_error(y_test, y_pred)

    # Calcul du coefficient de détermination R²
        r2 = r2_score(y_test, y_pred)

    # Affichage des métriques
        st.write("Score sur ensemble d'entraînement:", train_score)
        st.write("Score sur ensemble de test:", test_score)
        st.write("Mean Squared Error:", mse)
        st.write("R² Score:", r2)

    # Création de la figure Plotly
        fig = go.Figure()

    # Tracé des données réelles
        fig.add_trace(go.Scatter(x=Fusion['year'], y=Fusion[selected_variable], mode='markers', name='Données réelles', marker=dict(color='blue')))

    # Tracé de la régression polynomiale
        x_range = np.linspace(Fusion['year'].min(), Fusion['year'].max(), 100).reshape(-1, 1)
        y_poly = model_poly.predict(poly.transform(x_range))
        fig.add_trace(go.Scatter(x=x_range.flatten(), y=y_poly, mode='lines', name='Régression polynomiale', line=dict(color='red')))

    # Ajout des prédictions futures
        fig.add_trace(go.Scatter(x=df_predictions_poly['year'], y=df_predictions_poly['moyenne_annuelle_pred_poly'], mode='markers', name='Prédictions futures', marker=dict(color='green')))

    # Mise en forme du graphique
        fig.update_layout(title=f'Régression polynomiale des {selected_variable} en fonction du temps',
                    xaxis_title='Année',
                    yaxis_title=selected_variable,
                    showlegend=True,
                    template='plotly_white')

    # Affichage du graphique
        st.plotly_chart(fig)

    #--------------------------RESIDUS------------------------------------------------------------------------"

    # Charger les données
        Fusion = get_data_2()

    # Afficher le titre
        st.title('Résidus de la régression polynomiale')

    # Sélection de la variable
        selected_variable = st.selectbox('Sélectionner la variable à étudier pour les résidus', Fusion.columns)

    # Données d'entraînement
        X = Fusion['year'].values.reshape(-1, 1)
        y = Fusion[selected_variable]

    # Transformation polynomiale des features
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

    # Entraînement du modèle de régression polynomiale
        model_poly = LinearRegression()
        model_poly.fit(X_poly, y)

    # Prédictions sur les données d'entraînement
        y_pred = model_poly.predict(X_poly)

    # Calcul des résidus
        residuals = y - y_pred

    # Création du graphique des résidus
        fig_residus = go.Figure()

    # Tracé des résidus
        fig_residus.add_trace(go.Scatter(x=Fusion['year'], y=residuals, mode='markers', name='Résidus', marker=dict(color='blue')))

    # Mise en forme du graphique
        fig_residus.update_layout(title='Graphique des résidus de la régression polynomiale pour ' + selected_variable,
                            xaxis_title='Année',
                            yaxis_title='Résidus',
                            showlegend=True,
                            template='plotly_white')

    # Affichage du graphique dans Streamlit
        st.plotly_chart(fig_residus)

    # Affichage de la moyenne des résidus
        st.write('Moyenne des résidus :', residuals.mean())
        
if page == pages[4]:
    
    st.header('PAR ZONES')
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from plotly import graph_objs as go
    import plotly.express as px
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
    from sklearn.preprocessing import PolynomialFeatures

    zone=pd.read_csv("/Users/maison/Documents/PROJET/ZonAnn.Ts+dSST.csv")
    
    from PIL import Image
    from numpy import asarray
    img = Image.open('/Users/maison/Desktop/temperature-7164936_1280.png')
    numpydata = asarray(img)
    st.image(numpydata)
    
    
    if st.checkbox("EXPLORATION"):
    
        st.title("Analyse par zone géographique")
        st.write("### Introduction")
        st.dataframe(zone.head())
        st.write(zone.shape)
        st.dataframe(zone.describe())
        if st.checkbox("Afficher les NA"):
            st.write("Nombre de valeurs manquantes :", zone.isna().sum().sum())
        if st.checkbox("Afficher les doublons"):
            st.write("Nombre de doublons :", zone.duplicated().sum())

        new_names =  {'64N-90N' : 'Artic',
                    '44N-64N' : 'North_midlatitude',
                    '24N-44N' : 'North_Subtropical',
                    'EQU-24N' : 'North_Tropical',
                    '24S-EQU' : 'South_Tropical',
                    '44S-24S' : 'South_Subtropical',
                        '64S-44S' : 'South_midlatitude',
                            '90S-64S' : 'Antarctic'}

        zone = zone.rename(new_names, axis = 1)

        zoneglob = zone.loc[:, ['Year','Glob','NHem','SHem','24N-90N','24S-24N','90S-24S']]

        new_names =  {'24N-90N' : 'L24N_90N',
                '24S-24N' : 'L24S_24N',
                '90S-24S' : 'L90S_24S'}

        zoneglob = zoneglob.rename(new_names, axis = 1)

        zone= zone.loc[:, ['Year','Glob','NHem','SHem','Artic','North_midlatitude','North_Subtropical','North_Tropical','South_Tropical','South_Subtropical','South_midlatitude','Antarctic']]

        zone[144:12]= zone[144:12] / 100    
              
    if st.checkbox('DATA VISUALISATION'):
        st.write("### DataVizualization")
        
        def plot_temperature_variation(zoneglob):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=zoneglob['Year'], y=zoneglob['L24N_90N'],
                                     name='Zone Nord', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=zoneglob['Year'], y=zoneglob['Glob'],
                                     name='Globale', line=dict(color='grey')))
            fig.add_trace(go.Scatter(x=zoneglob['Year'], y=zoneglob['L24S_24N'],
                                     name='Zone Equateur', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=zoneglob['Year'], y=zoneglob['L90S_24S'],
                                     name='Zone Sud', line=dict(color='blue')))

            fig.update_layout(legend_title='Zones')
            fig.update_layout(title='Les variations de température par large zone géographique depuis 1880',
                            xaxis_title='Année',
                            yaxis_title='Variations de température')

            return fig
        new_names =  {'64N-90N' : 'Artic',
                    '44N-64N' : 'North_midlatitude',
                    '24N-44N' : 'North_Subtropical',
                    'EQU-24N' : 'North_Tropical',
                    '24S-EQU' : 'South_Tropical',
                    '44S-24S' : 'South_Subtropical',
                        '64S-44S' : 'South_midlatitude',
                            '90S-64S' : 'Antarctic'}
        zone = zone.rename(new_names, axis = 1)

        zoneglob = zone.loc[:, ['Year','Glob','NHem','SHem','24N-90N','24S-24N','90S-24S']]

        new_names =  {'24N-90N' : 'L24N_90N',
                '24S-24N' : 'L24S_24N',
                '90S-24S' : 'L90S_24S'}

        zoneglob = zoneglob.rename(new_names, axis = 1)

        zone= zone.loc[:, ['Year','Glob','NHem','SHem','Artic','North_midlatitude','North_Subtropical','North_Tropical','South_Tropical','South_Subtropical','South_midlatitude','Antarctic']]

        fig = plot_temperature_variation(zoneglob)
        st.plotly_chart(fig)

        def plot_temperature_variation(zone):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=zone['Year'], y=zone['Artic'], name='Artic'))
            fig.add_trace(go.Scatter(x=zone['Year'], y=zone['North_midlatitude'], name='North_midlatitude'))
            fig.add_trace(go.Scatter(x=zone['Year'], y=zone['North_Subtropical'], name='North_Subtropical'))
            fig.add_trace(go.Scatter(x=zone['Year'], y=zone['North_Tropical'], name='North_Tropical'))
            fig.add_trace(go.Scatter(x=zone['Year'], y=zone['South_Tropical'], name='South_Tropical'))
            fig.add_trace(go.Scatter(x=zone['Year'], y=zone['South_Subtropical'], name='South_Subtropical'))
            fig.add_trace(go.Scatter(x=zone['Year'], y=zone['South_midlatitude'], name='South_midlatitude'))
            fig.add_trace(go.Scatter(x=zone['Year'], y=zone['Antarctic'], name='Antarctic'))
            fig.update_layout(legend_title='Zones:')
            fig.update_layout(title='Les variations de température par zone géographique réduite depuis 1880',
                              xaxis_title='Année',
                              yaxis_title='Variations de température',
                              legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="right", x=1)
                              )
            return fig
        fig_zone = plot_temperature_variation(zone)
        st.plotly_chart(fig_zone)
        
        Co2=pd.read_csv("/Users/maison/Documents/PROJET/owid-co2-data_Nathan.csv")
        co = Co2[['country','year','iso_code','population','gdp','co2','co2_including_luc','consumption_co2','cumulative_co2','gas_co2','share_global_co2']]
        co = co.reset_index(drop=True)
        co = co.loc[(co.isna().sum(axis=1)) < 6]
        word = Co2[['country','year','co2','co2_including_luc','consumption_co2','cumulative_co2','gas_co2','share_global_co2']] 
        co = co.sort_values('year', ascending=True)
        co['co2'] = co['co2'].replace(0, np.nan)

        def plot_co2_choropleth(co):
            fig = px.choropleth(co, locations="iso_code",
                                color="co2", hover_name="country",
                                animation_frame='year',
                                color_continuous_scale=px.colors.sequential.Plasma,
                                range_color=[50, 5500],
                                projection="equirectangular",
                                title="L'augmentation du CO2 au niveau mondial par pays")
    
            fig.update_geos(lataxis_showgrid=True)

            return fig
        
        fig_co2_choropleth = plot_co2_choropleth(co)
        st.plotly_chart(fig_co2_choropleth)
        
    
        Co2=pd.read_csv("/Users/maison/Documents/PROJET/owid-co2-data_Nathan.csv")
        co = Co2[['country','year','co2_including_luc']]
        valeur_a_garder='World'
        dfworld = co.loc[co['country']==valeur_a_garder]
        dfworld = dfworld.loc[dfworld['year'] > 1880]
        dfworld=dfworld.sort_values(by = 'year', ascending=False)
        dfworld=dfworld.reset_index(drop=True)
        zonepred=zone.sort_values(by = 'Year', ascending=False)
        zonepred=zonepred.reset_index(drop=True)
        merged_df = pd.merge(dfworld, zonepred, left_on='year', right_on='Year')
        merged_df = merged_df.drop('Year', axis=1)
        merged_df.drop('country', axis=1, inplace=True)
        merged_df=merged_df.sort_values(by = 'year', ascending=True)
        merged_df= merged_df.fillna(method='ffill')
        merged_df=merged_df.sort_values(by = 'year', ascending=False)
        merged_df = merged_df.loc[merged_df['year'] > 1881]
        df = merged_df
        
    if st.checkbox('MODELISATION'):   
        
        new_names =  {'64N-90N' : 'Artic',
                    '44N-64N' : 'North_midlatitude',
                    '24N-44N' : 'North_Subtropical',
                    'EQU-24N' : 'North_Tropical',
                    '24S-EQU' : 'South_Tropical',
                    '44S-24S' : 'South_Subtropical',
                        '64S-44S' : 'South_midlatitude',
                            '90S-64S' : 'Antarctic'}
        zone = zone.rename(new_names, axis = 1)

        zoneglob = zone.loc[:, ['Year','Glob','NHem','SHem','24N-90N','24S-24N','90S-24S']]

        new_names =  {'24N-90N' : 'L24N_90N',
                '24S-24N' : 'L24S_24N',
                '90S-24S' : 'L90S_24S'}

        zoneglob = zoneglob.rename(new_names, axis = 1)

        zone= zone.loc[:, ['Year','Glob','NHem','SHem','Artic','North_midlatitude','North_Subtropical','North_Tropical','South_Tropical','South_Subtropical','South_midlatitude','Antarctic']]

        
        Co2=pd.read_csv("/Users/maison/Documents/PROJET/owid-co2-data_Nathan.csv")
        co = Co2[['country','year','co2_including_luc']]
        valeur_a_garder='World'
        dfworld = co.loc[co['country']==valeur_a_garder]
        dfworld = dfworld.loc[dfworld['year'] > 1880]
        dfworld=dfworld.sort_values(by = 'year', ascending=False)
        dfworld=dfworld.reset_index(drop=True)
        zonepred=zone.sort_values(by = 'Year', ascending=False)
        zonepred=zonepred.reset_index(drop=True)
        merged_df = pd.merge(dfworld, zonepred, left_on='year', right_on='Year')
        merged_df = merged_df.drop('Year', axis=1)
        merged_df.drop('country', axis=1, inplace=True)
        merged_df=merged_df.sort_values(by = 'year', ascending=True)
        merged_df= merged_df.fillna(method='ffill')
        merged_df=merged_df.sort_values(by = 'year', ascending=False)
        merged_df = merged_df.loc[merged_df['year'] > 1881]
        df = merged_df
        
        st.write("### Modélisation")
        X = df[['year']]
        y = df['Artic']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
        selected_model = st.selectbox("Sélectionnez le modèle", ["Régression linéaire", "Arbre de décision", "Random Forest"])
        if selected_model == "Régression linéaire":model = LinearRegression()
        elif selected_model == "Arbre de décision":model = DecisionTreeRegressor()
        elif selected_model == "Random Forest":model = RandomForestRegressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score_train = model.score(X_train, y_train)
        score_test = model.score(X_test, y_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        st.write(f"Score train {selected_model.lower()} : {score_train}")
        st.write(f"Score test {selected_model.lower()} : {score_test}")
        st.write("Mean Squared Error:", mse)
        st.write("R^2 Score:", r2)
        df_predictions = pd.DataFrame({'year': X_test.squeeze(), 'Prédictions': predictions.squeeze()})
        df_sorted = df_predictions.sort_values(by='year')
        X_test_sorted = df_sorted['year'].values
        predictions_sorted = df_sorted['Prédictions'].values
        plt.scatter(X, y, color='blue', label='Données réelles')
        plt.plot(X_test_sorted, predictions_sorted, color='red', linewidth=2, label=f'Régression {selected_model.lower()}')
        plt.xlabel('Année')
        plt.ylabel('Anomalie de température')
        plt.title(f'Regression {selected_model.lower()} des anomalies de température')
        plt.legend()
        st.pyplot(plt)
            

    if st.checkbox('PREDICTIONS'):
         new_names =  {'64N-90N' : 'Artic',
                     '44N-64N' : 'North_midlatitude',
                     '24N-44N' : 'North_Subtropical',
                     'EQU-24N' : 'North_Tropical',
                     '24S-EQU' : 'South_Tropical',
                     '44S-24S' : 'South_Subtropical',
                         '64S-44S' : 'South_midlatitude',
                             '90S-64S' : 'Antarctic'}
         zone = zone.rename(new_names, axis = 1)

         zoneglob = zone.loc[:, ['Year','Glob','NHem','SHem','24N-90N','24S-24N','90S-24S']]

         new_names =  {'24N-90N' : 'L24N_90N',
                 '24S-24N' : 'L24S_24N',
                 '90S-24S' : 'L90S_24S'}

         zoneglob = zoneglob.rename(new_names, axis = 1)

         zone= zone.loc[:, ['Year','Glob','NHem','SHem','Artic','North_midlatitude','North_Subtropical','North_Tropical','South_Tropical','South_Subtropical','South_midlatitude','Antarctic']]

         Co2=pd.read_csv("/Users/maison/Documents/PROJET/owid-co2-data_Nathan.csv")
         co = Co2[['country','year','co2_including_luc']]
         valeur_a_garder='World'
         dfworld = co.loc[co['country']==valeur_a_garder]
         dfworld = dfworld.loc[dfworld['year'] > 1880]
         dfworld=dfworld.sort_values(by = 'year', ascending=False)
         dfworld=dfworld.reset_index(drop=True)
         zonepred=zone.sort_values(by = 'Year', ascending=False)
         zonepred=zonepred.reset_index(drop=True)
         merged_df = pd.merge(dfworld, zonepred, left_on='year', right_on='Year')
         merged_df = merged_df.drop('Year', axis=1)
         merged_df.drop('country', axis=1, inplace=True)
         merged_df=merged_df.sort_values(by = 'year', ascending=True)
         merged_df= merged_df.fillna(method='ffill')
         merged_df=merged_df.sort_values(by = 'year', ascending=False)
         merged_df = merged_df.loc[merged_df['year'] > 1881]
         df = merged_df
         
         st.title("Prédiction des variations de température par zone géographique")
         def main():
             st.title(" ")
         region_options = ["Arctique", "Latitude moyenne nord",
                      "Nord subtropicale", "Nord tropicale", "Sud tropicale","Sud subtropicale","Latitude moyenne sud","Antarctique"]
         region = st.radio("Selectionner Region géographique", region_options)
         if region == "Arctique":
             st.title("Arctique")
             y_column = 'Artic'
         elif region == "Antarctique":
             st.title("Antarctique")
             y_column = 'Antarctic'
         elif region == "Latitude moyenne nord":
             st.title("Latitude moyenne nord")
             y_column = 'North_midlatitude'
         elif region == "Latitude moyenne sud":
             st.title("Latitude moyenne sud")
             y_column = 'South_midlatitude'
         elif region == "Nord subtropicale":
             st.title("Nord subtropicale")
             y_column = 'North_Subtropical'
         elif region == "Sud subtropicale":
             st.title("Sud subtropicale")
             y_column = 'South_Subtropical'
         elif region == "Nord tropicale":
             st.title("Nord tropicale")
             y_column = 'North_Tropical'
         elif region == "Sud tropicale":
             st.title("Sud tropicale")
             y_column = 'South_Tropical'
        
         X = df[['year']]
         y = df[y_column]
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
         poly_features = PolynomialFeatures(degree=2)
         X_train_poly = poly_features.fit_transform(X_train)
         X_test_poly = poly_features.transform(X_test)
         model = LinearRegression()
         model.fit(X_train_poly, y_train)
         years_to_predict = np.array([[2024], [2025], [2026]])
         years_to_predict_poly = poly_features.transform(years_to_predict)
         predictions = model.predict(years_to_predict_poly)
         for year, prediction in zip(years_to_predict.flatten(), predictions):
             st.write(f"Année {year}: Prédiction de la variation de temperature = {prediction:.2f}°C")
         y_pred = model.predict(X_test_poly)
         mse = mean_squared_error(y_test, y_pred)
         st.subheader("Evaluation du modele")
         st.write(f"Mean Squared Error: {mse:.2f}")
         plot_data = pd.DataFrame({'year': X_test.values.flatten(), 'Predictions': y_pred.flatten()})
         plot_data_sorted = plot_data.sort_values(by='year')
         fig, ax = plt.subplots()
         ax.scatter(X, y, color='blue', label='Actual Data')
         ax.plot(plot_data_sorted['year'], plot_data_sorted['Predictions'], color='red', linewidth=2, label='Linear Regression')
         ax.scatter([2024, 2025, 2026], predictions, color='green', label='Predictions 2024, 2025, 2026')
         ax.set_xlabel('Année')
         ax.set_ylabel('Variation de Temperature')
         ax.set_title('Regression avec Polynomial Features')
         st.pyplot(fig)
         if __name__ == "__main__":
             main()

if page == pages[5]:
    st.header('Conclusion')
    from PIL import Image
    from numpy import asarray
    img = Image.open('/Users/maison/Downloads/earth-hour-4776711_1280.jpg')
    numpydata = asarray(img)
    st.image(numpydata, use_column_width= True)
    
    
    st.write('Dans le cadre de l\'objectif qui nous était donné de constater le dérèglement climatique, nous pouvons attester d’une évolution des températures qui augmentent au fil des ans tel que prédit par les modèles et quel que soit l’hémisphère ou la zone géographique.')
    st.write('Une tendance à l\'accéleration de ces augmentations a été mise en évidence à la comparaison de différentes périodes.')
    st.write('A l\'analyse des données sur le CO2, nous avons remarqué que cette tendance est significativement corrélée à l\'augmentation des émissions de CO2.')
    st.write('Il est important de noter que les modèles de régression seront des outils efficaces car ils pourront transmettre des données essentielles pour comprendre et prédire les tendances climatiques et ainsi aider à prendre des décisions éclairées pour l’avenir.')
