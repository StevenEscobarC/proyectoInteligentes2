from bson import ObjectId
from flask import Flask, request, jsonify
import os
from datetime import datetime

import numpy as np
from Controladores.ControladorModelo import ControladorModelo
import json
import pandas as pd
import pymongo
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import joblib  
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from flask_cors import CORS
import seaborn as sns
from sklearn.decomposition import PCA
#from keras.models import load_model
#from tensorflow import keras
        
app = Flask(__name__)


cors = CORS(app)

client = pymongo.MongoClient("mongodb+srv://steven:frZT5fK3llyw5RVY@proyectointeligentes2.qyb4k8w.mongodb.net/?retryWrites=true&w=majority")
database = client["Inteligentes2"]
coleccion_datos = database["DataSets"]
coleccion_datos_imputados = database["DataSetsImputados"]
coleccion_modelos = database["Modelos"]
coleccion_codigos = database["Codigos"]
coleccion_datos_PCA = database["DataPCA"]
coleccion_entrenamientos = database["Entrenamientos"]
df = "" 
filename = ""
X_train, X_test, y_train, y_test = [], [], [], []
y_pred = []

@app.route('/load', methods=['POST'])
def load_file():
    file = request.files.get('file') 
    if file:
        global filename
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + file.filename
        file.save(os.path.join('Archivos/', filename))
        global df
        try:
            df = pd.read_excel(file)
        except pd.errors.EmptyDataError:
            return jsonify({'error': 'El archivo Excel está vacío.'}), 400
        except pd.errors.ParserError as pe:
            return jsonify({'error': f'Error parsing Excel file: {pe}'}), 400
        
        file_json = df.to_dict('records')
        
        file_doc = {
            'documento': filename,
            'contenido': file_json,
            'upload_time': datetime.now(),
        }
        result = coleccion_datos.insert_one(file_doc)
        id_insertado = str(result.inserted_id)

        columnas = df.columns
        print(columnas.size)
        
        originales=df.copy()
        for columna in columnas:
            if df[columna].dtypes == 'object' or df[columna].dtypes == 'bool':
                labelencoder_X = LabelEncoder()
                df[columna] =  labelencoder_X.fit_transform(df[columna])
        aux=[]
        lista=[]
        dicc={}
        # se guardan los codigos para la decodificacion
        for columna in columnas:
            if originales[columna].dtypes == 'object' or originales[columna].dtypes == 'bool':
                for valorCod,valorOg in zip(df[columna],originales[columna]):
                    if valorCod not in aux:
                        new={}
                        new["valorCod"]=valorCod
                        new["valorOg"]=valorOg
                        lista.append(new)
                        aux.append(valorCod)
                aux=[]
                dicc[columna]=lista
                lista=[]
        file_doc = {"documento":filename, "codigos": dicc}
        coleccion_codigos.insert_one(file_doc)

        return jsonify({'message': 'Archivo guardado exitosamente.','nombre':filename, 'id_insertado':id_insertado}), 200
    else:
        return jsonify({'error': 'No se recibió ningún archivo.'}), 400
    
# @app.route('/basics_statistics/<dataset_id>', methods=['GET'])
# def basics_statistics(dataset_id):
#     try:
#         object_id = ObjectId(dataset_id)
#         resultados = coleccion_datos.find({ '_id': object_id })
#         lista_resultados = []
#         for documento in resultados:
#             documento = convertir_a_cadena(documento)
#             lista_resultados.append(documento['contenido'])
#         json_resultados = json.dumps(lista_resultados)
#         return json_resultados, 200
#     except AttributeError:
#         return jsonify({'error': 'No se ha encontrado el dataset.'}), 400
#     except Exception:
#         return jsonify({'error': 'Ha ocurrido un error encontrando el dataset.'}), 400
    
@app.route('/basics_statistics/<registro_id>', methods=['GET'])
def get_basics_statistics(registro_id):
    try:
        # Convertir el ID del registro a ObjectId de MongoDB
        object_id = ObjectId(registro_id)
        
        # Buscar el registro por ID
        registro = coleccion_datos.find_one({'_id': object_id})

        if registro:
            # Obtener el contenido del archivo en formato JSON
            json_content = registro['contenido']

            # Convertir el JSON a un DataFrame de Pandas
            global df

            df = pd.DataFrame(json_content)
            

            df_describe = df.describe().to_dict()

            return jsonify({'message': 'Dataset obtenido exitosamente.', 'documento obtenido:': registro['documento'], 'basic_statistics': df_describe}), 200
        else:
            registro_impu = coleccion_datos_imputados.find_one({'_id': object_id})
            print(registro_impu)
            if registro_impu:
                 # Obtener el contenido del archivo en formato JSON
                json_content = registro_impu['contenido']

                df = pd.DataFrame(json_content)
                

                df_describe = df.describe().to_dict()

                return jsonify({'message': 'Dataset imputado obtenido exitosamente.', 'documento obtenido:': registro_impu['documentoSinImputarCod'], 'basic_statistics': df_describe}), 200
            else: 
                return jsonify({'error': 'Registro no encontrado.'}), 404

    except Exception as e:
        print(e)
        return jsonify({'error': 'Ha ocurrido un error.'}), 500
    
@app.route('/columns_describe/<registro_id>', methods=['GET'])
def columns_describe(registro_id):
    try:
        # Convertir el ID del registro a ObjectId de MongoDB
        object_id = ObjectId(registro_id)
        
        # Buscar el registro por ID
        registro = coleccion_datos.find_one({'_id': object_id})

        if registro:
            # Obtener el contenido del archivo en formato JSON
            json_content = registro['contenido']

            # Convertir el JSON a un DataFrame de Pandas
            global df

            df = pd.DataFrame(json_content)
            

            df_dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
        return jsonify({'message': 'Dataset descrito correctamente.', 'columns_describe': df_dtypes}), 200
    except Exception as e:
        print(e)
        return jsonify({'error': 'Ha ocurrido un error.'}), 500
    


@app.route('/imputation/<registro_id>/<number_type>', methods=['GET'])
def columns(registro_id,number_type):
    try:
        print(number_type,' registro ',registro_id)
        # Convertir el ID del registro a ObjectId de MongoDB
        object_id = ObjectId(registro_id)
        
        # Buscar el registro por ID
        registro = coleccion_datos.find_one({'_id': object_id})
        

        if registro:
            # Obtener el contenido del archivo en formato JSON
            json_content = registro['contenido']

            # Convertir el JSON a un DataFrame de Pandas
            global df

            df = pd.DataFrame(json_content)
            #originales = df.copy()
            imputacion = None
            #el tipo uno elimina datos faltantes y el tipo 2, reemplaza por la media los valors numericos y la moda por los categoricos
            if number_type=="1":
                df=df.dropna()
                imputacion = "Eliminacion de datos faltantes"
                
            elif number_type=="2":
                
                # df=df.fillna(df.mean())
                # df=df.fillna(df.mode().iloc[0])
                # imputacion = "Reemplazo por la media y la moda"
                # Identify numeric and categorical columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                categorical_cols = df.select_dtypes(include=['object']).columns

                # Impute numeric columns with mean
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

                # Impute categorical columns with mode
                df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
                imputacion = "Reemplazo por la media y la moda"

        

        file_imputados = {'documentoSinImputarCod': registro_id, 'contenido': df.to_dict('records'), 'upload_time': datetime.now(), 'Tipo imputacion': imputacion}
        coleccion_datos_imputados.insert_one(file_imputados)

        return jsonify({'message': 'Dataset imputado correctamente.', 'Tipo de imputación aplicada': imputacion}), 200
    except Exception as e:
        print(e)
        return jsonify({'error': 'Ha ocurrido un error.'}), 500

@app.route('/general_univariate_graphs/<dataset_id>', methods=['POST'])
def general_univariate_graphs(dataset_id):
    try:
        # Convertir el ID del registro a ObjectId de MongoDB
        #object_id = ObjectId(dataset_id)
        
        # Buscar el registro por ID
        registro = coleccion_datos_imputados.find_one({'documentoSinImputarCod': dataset_id})

        if registro:
            # Obtener el contenido del archivo en formato JSON
            json_content = registro['contenido']

            # Convertir el JSON a un DataFrame de Pandas
            global df
            df = pd.DataFrame(json_content)
            for columna in df.columns:
                if df[columna].dtypes == 'object' or df[columna].dtypes == 'bool':
                    labelencoder_x = LabelEncoder()
                    df[columna] =  labelencoder_x.fit_transform(df[columna])
            
            # Crear una carpeta para almacenar los gráficos (si no existe)
            carpeta_graficos = os.path.join('Graficos/', dataset_id)
            os.makedirs(carpeta_graficos, exist_ok=True)

            rutas_graficos = []  # Para almacenar las rutas de los gráficos generados

            for columna in df.columns:
                carpeta_columna = os.path.join(carpeta_graficos, columna)
                os.makedirs(carpeta_columna, exist_ok=True)

                # Histograma
                plt.figure()
                sns.histplot(df[columna], kde=True)
                ruta_histograma = os.path.join(carpeta_columna, f"{columna}_histograma.png")
                plt.savefig(ruta_histograma)
                rutas_graficos.append(ruta_histograma)
                plt.close()

                # Diagrama de caja (solo para variables numéricas)
                if pd.api.types.is_numeric_dtype(df[columna]):
                    plt.figure()
                    sns.boxplot(x=df[columna])  # Elimina la columna x para que el eje x sea automático
                    ruta_diagrama_caja = os.path.join(carpeta_columna, f"{columna}_diagrama_caja.png")
                    plt.savefig(ruta_diagrama_caja)
                    rutas_graficos.append(ruta_diagrama_caja)
                    plt.close()

                # Análisis de distribución de probabilidad (Kernel Density Estimate - KDE)
                plt.figure()
                sns.kdeplot(df[columna], fill=True)
                ruta_distribucion_probabilidad = os.path.join(carpeta_columna, f"{columna}_distribucion_probabilidad.png")
                plt.savefig(ruta_distribucion_probabilidad)
                rutas_graficos.append(ruta_distribucion_probabilidad)
                plt.close()

             

            return jsonify({'message': 'Gráficos generados exitosamente.', 'rutas_graficos': rutas_graficos}), 200
        else:
            return jsonify({'error': 'Registro no encontrado.'}), 404

    except Exception as e:
        print(e)
        return jsonify({'error': 'Ha ocurrido un error generando los gráficos.'}), 500
    
# Función para crear y guardar gráficos de diagramas de caja y gráficos de densidad por clase
def create_and_save_plots(df, column_name, dataset_id):
    # Crear una carpeta para almacenar los gráficos (si no existe)
    carpeta_graficos = os.path.join('GraficosClase/', dataset_id)
    os.makedirs(carpeta_graficos, exist_ok=True)

    # Crear diagrama de caja
    plt.figure()
    sns.boxplot(x='class', y=column_name, data=df)
    ruta_diagrama_caja = os.path.join(carpeta_graficos, f"{column_name}_boxplot.png")
    plt.savefig(ruta_diagrama_caja)
    plt.close()

    # Crear gráfico de densidad
    plt.figure()
    sns.kdeplot(x=column_name, hue='class', data=df, fill=True)
    ruta_grafico_densidad = os.path.join(carpeta_graficos, f"{column_name}_density_plot.png")
    plt.savefig(ruta_grafico_densidad)
    plt.close()

    return ruta_diagrama_caja, ruta_grafico_densidad

@app.route('/univariate_graphs_class/<dataset_id>/', methods=['POST'])
def univariate_graphs_class(dataset_id):
    try:
        registro = coleccion_datos_imputados.find_one({'documentoSinImputarCod': dataset_id})

        if registro:
            # Obtener el contenido del archivo en formato JSON
            json_content = registro['contenido']

            # Convertir el JSON a un DataFrame de Pandas
            global df
            df = pd.DataFrame(json_content)
            for columna in df.columns:
                if df[columna].dtypes == 'object' or df[columna].dtypes == 'bool':
                    labelencoder_x = LabelEncoder()
                    df[columna] =  labelencoder_x.fit_transform(df[columna])

            # Verificar si la columna y 'class' está presente en el DataFrame
            if 'class' not in df.columns:
                return jsonify({'error': 'La columna "class" no está presente en el conjunto de datos.'}), 400

            # Iterar sobre las columnas (excepto 'class') y crear y almacenar los gráficos
            rutas_graficos = {}
            for columna in df.columns:
                if columna != 'class':
                    ruta_diagrama_caja, ruta_grafico_densidad = create_and_save_plots(df, columna, dataset_id)
                    rutas_graficos[columna] = {'boxplot': ruta_diagrama_caja, 'density_plot': ruta_grafico_densidad}

            return jsonify({'message': 'Gráficos caja y densidad por clase generados y almacenados exitosamente.', 'rutas_graficos': rutas_graficos}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': 'Ha ocurrido un error generando los gráficos.'}), 500
    
# Función para crear y guardar el pair plot
def create_and_save_pair_plot(df, dataset_id):
    # Crear una carpeta para almacenar los gráficos (si no existe)
    carpeta_graficos = os.path.join('GraficosBivariate/', dataset_id)
    os.makedirs(carpeta_graficos, exist_ok=True)

    # Crear el pair plot
    plt.figure()
    #sns.pairplot(df)
    sns.pairplot(df, hue = "class",
             corner = True)
    ruta_pair_plot = os.path.join(carpeta_graficos, 'pair_plot.png')
    plt.savefig(ruta_pair_plot)
    plt.close()

    return ruta_pair_plot

@app.route('/bivariate_graphs_class/<dataset_id>/', methods=['GET'])
def bivariate_graphs_class(dataset_id):
    try:
        registro = coleccion_datos_imputados.find_one({'documentoSinImputarCod': dataset_id})

        if registro:
            # Obtener el contenido del archivo en formato JSON
            json_content = registro['contenido']

            # Convertir el JSON a un DataFrame de Pandas
            global df
            df = pd.DataFrame(json_content)
            for columna in df.columns:
                if df[columna].dtypes == 'object' or df[columna].dtypes == 'bool':
                    labelencoder_x = LabelEncoder()
                    df[columna] =  labelencoder_x.fit_transform(df[columna])

            # Verificar si la columna 'class' está presente en el DataFrame
            if 'class' not in df.columns:
                return jsonify({'error': 'La columna "class" no está presente en el conjunto de datos.'}), 400

            # Crear y guardar el pair plot
            ruta_pair_plot = create_and_save_pair_plot(df, dataset_id)

            return jsonify({'message': 'Pair plot generado y almacenado exitosamente.', 'ruta_pair_plot': ruta_pair_plot}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': 'Ha ocurrido un error generando el pair plot.'}), 500
    

# Función para crear y guardar el gráfico de correlación
def create_and_save_correlation_plot(df, dataset_id):
    # Crear una carpeta para almacenar los gráficos (si no existe)
    carpeta_graficos = os.path.join('GraficosMultivariateClass/', dataset_id)
    os.makedirs(carpeta_graficos, exist_ok=True)

    # Seleccionar columnas numéricas para el gráfico de correlación
    columnas_numericas = df.select_dtypes(include=['number']).columns

    # Crear el gráfico de correlación con un tamaño más grande
    plt.figure(figsize=(12, 10))  # Ajusta el tamaño de la figura según tus necesidades
    matriz_correlacion = df[columnas_numericas].corr()
    sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', fmt=".2f")
    ruta_correlation_plot = os.path.join(carpeta_graficos, 'correlation_plot.png')
    plt.savefig(ruta_correlation_plot)
    plt.close()

    return ruta_correlation_plot


@app.route('/multivariate_graphs_class/<dataset_id>/', methods=['GET'])
def multivariate_graphs_class(dataset_id):
    try:
        registro = coleccion_datos_imputados.find_one({'documentoSinImputarCod': dataset_id})

        if registro:
            # Obtener el contenido del archivo en formato JSON
            json_content = registro['contenido']

            # Convertir el JSON a un DataFrame de Pandas
            global df
            df = pd.DataFrame(json_content)
            for columna in df.columns:
                if df[columna].dtypes == 'object' or df[columna].dtypes == 'bool':
                    labelencoder_x = LabelEncoder()
                    df[columna] =  labelencoder_x.fit_transform(df[columna])

            # Verificar si la columna 'class' está presente en el DataFrame
            if 'class' not in df.columns:
                return jsonify({'error': 'La columna "class" no está presente en el conjunto de datos.'}), 400

            # Crear y guardar el gráfico de correlación
            ruta_correlation_plot = create_and_save_correlation_plot(df, dataset_id)

            return jsonify({'message': 'Gráfico de correlación generado y almacenado exitosamente.', 'ruta_correlation_plot': ruta_correlation_plot}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': 'Ha ocurrido un error generando el gráfico de correlación.'}), 500
    

#pca
# Función para aplicar PCA y retornar los pesos de las componentes y el nuevo dataset
def apply_pca(df, dataset_id, porcentaje_varianza=0.95):
    # Seleccionar solo columnas numéricas para PCA
    columnas_numericas = df.select_dtypes(include=['number']).columns
    data_numeric = df[columnas_numericas]

    # Aplicar PCA
    pca = PCA()
    pca_resultados = pca.fit_transform(data_numeric)

    # Determinar el número óptimo de componentes
    n_componentes_optimo = find_optimal_components(pca.explained_variance_ratio_, porcentaje_varianza)

    # Aplicar PCA con el número óptimo de componentes
    pca = PCA(n_components=n_componentes_optimo)
    pca_resultados = pca.fit_transform(data_numeric)

    # Crear un DataFrame con los resultados de PCA
    columnas_pca = [f'Componente_{i+1}' for i in range(n_componentes_optimo)]
    df_pca = pd.DataFrame(data=pca_resultados, columns=columnas_pca)

    # Agregar las columnas no numéricas al nuevo DataFrame
    df_pca = pd.concat([df[[col for col in df.columns if col not in columnas_numericas]], df_pca], axis=1)

    # Guardar el nuevo dataset con datos transformados
    carpeta_datasets_pca = os.path.join('DatasetsPCA/', dataset_id)
    os.makedirs(carpeta_datasets_pca, exist_ok=True)
    ruta_nuevo_dataset = os.path.join(carpeta_datasets_pca, 'dataset_pca.csv')
    df_pca.to_csv(ruta_nuevo_dataset, index=False)

    # Retornar los resultados de PCA y la ruta del nuevo dataset
    return {'pesos_componentes': pca.components_.tolist(), 'ruta_nuevo_dataset': ruta_nuevo_dataset}

# Función para aplicar PCA y crear un nuevo CSV solo con las componentes PCA y nombres originales de características
def apply_pca_and_create_csv(df, dataset_id, porcentaje_varianza=0.95):
    # Seleccionar solo columnas numéricas para PCA
    columnas_numericas = df.select_dtypes(include=['number']).columns
    data_numeric = df[columnas_numericas]

    # Aplicar PCA
    pca = PCA()
    pca_resultados = pca.fit_transform(data_numeric)

    # Determinar el número óptimo de componentes
    n_componentes_optimo = find_optimal_components(pca.explained_variance_ratio_, porcentaje_varianza)

    # Aplicar PCA con el número óptimo de componentes
    pca = PCA(n_components=n_componentes_optimo)
    pca_resultados = pca.fit_transform(data_numeric)

    # Crear un DataFrame solo con las componentes PCA
    columnas_pca = [f'Componente_{i+1}' for i in range(n_componentes_optimo)]
    df_pca = pd.DataFrame(data=pca_resultados, columns=columnas_pca)

    # Agregar las columnas no numéricas al nuevo DataFrame
    df_pca = pd.concat([df[[col for col in df.columns if col not in columnas_numericas]], df_pca], axis=1)

    # Guardar el nuevo dataset con datos transformados
    carpeta_datasets_pca = os.path.join('DatasetsPCA/', dataset_id)
    os.makedirs(carpeta_datasets_pca, exist_ok=True)
    ruta_nuevo_dataset = os.path.join(carpeta_datasets_pca, 'dataset_pca.csv')
    df_pca.to_csv(ruta_nuevo_dataset, index=False)

    # Retornar la ruta del nuevo dataset
    return {'ruta_nuevo_dataset': ruta_nuevo_dataset}

# Función para encontrar el número óptimo de componentes que explican el porcentaje de varianza deseado
def find_optimal_components(explained_variance_ratio, target_variance):
    cum_var_ratio = 0.0
    n_components = 0

    for var_ratio in explained_variance_ratio:
        cum_var_ratio += var_ratio
        n_components += 1

        if cum_var_ratio >= target_variance:
            break

    return n_components

@app.route('/pca/<dataset_id>/', methods=['POST'])
def pca(dataset_id):
    try:
        registro = coleccion_datos_imputados.find_one({'documentoSinImputarCod': dataset_id})

        if registro:
            # Obtener el contenido del archivo en formato JSON
            json_content = registro['contenido']

            # Convertir el JSON a un DataFrame de Pandas
            global df
            df = pd.DataFrame(json_content)
            for columna in df.columns:
                if df[columna].dtypes == 'object' or df[columna].dtypes == 'bool':
                    labelencoder_x = LabelEncoder()
                    df[columna] =  labelencoder_x.fit_transform(df[columna])

            # Verificar si el DataFrame contiene al menos dos columnas numéricas
            columnas_numericas = df.select_dtypes(include=['number']).columns
            if len(columnas_numericas) < 2:
                return jsonify({'error': 'Se requieren al menos dos columnas numéricas para aplicar PCA.'}), 400

            # Aplicar PCA y obtener resultados
            resultados_pca = apply_pca_and_create_csv(df, dataset_id)

            return jsonify({'message': 'PCA aplicado exitosamente.', 'resultados_pca': resultados_pca}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': 'Ha ocurrido un error aplicando PCA.'}), 500
######################################################################################################
    
@app.route('/train/<dataset_id>/', methods=['POST'])
def train(dataset_id):
    try:
        # Convertir el ID del registro a ObjectId de MongoDB
        # object_id = ObjectId(dataset_id)

        # Buscar el registro por ID
        registro = coleccion_datos_imputados.find_one({'documentoSinImputarCod': dataset_id})

        if registro:
            # Obtener el contenido del archivo en formato JSON
            json_content = registro['contenido']

            # Convertir el JSON a un DataFrame de Pandas
            df = pd.DataFrame(json_content)

            for columna in df.columns:
                if df[columna].dtypes == 'object' or df[columna].dtypes == 'bool':
                    labelencoder_x = LabelEncoder()
                    df[columna] =  labelencoder_x.fit_transform(df[columna])

            # Separar variables predictoras (x) y variable objetivo (y)
            x = df.drop('class', axis=1)
            y = df['class']

            # Validar la solicitud JSON
            body = request.get_json()
            if 'normalization' not in body or 'option_train' not in body or 'algorithms' not in body:
                return jsonify({'error': 'La solicitud JSON no contiene la información necesaria.'}), 400

            print(df.columns, ' antes')


            # for columna in df.select_dtypes(include=['object', 'bool']).columns:
            #     df = pd.concat([df, pd.get_dummies(df[columna], prefix=columna)], axis=1)
            #     df.drop(columna, axis=1, inplace=True)

            

            normalization = body['normalization']
            option_train = body['option_train']
            algorithms = body['algorithms']

            global X_train, X_test, y_train, y_test

            if option_train == 1:  # Hold out
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
            elif option_train == 2:  # Cross Validation
                X_train, X_test, y_train, y_test = x, x, y, y

            if normalization == 1:  # MinMax
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                print('entra scaler despues')
            elif normalization == 2:  # Standard Scaler
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            # Entrenar modelos para cada algoritmo seleccionado
            routes = []
            nombres_algoritmos = []
            tasks = []

            for algorithm_id in algorithms:
                nombre, model = ControladorModelo().entrenar(algorithm_id)
                model.fit(X_train, y_train)
                global y_pred
                
                y_pred = model.predict(X_test)
                print(y_pred)
                nombres_algoritmos.append(nombre)

                # Modificar la variable 'route' para incluir el nombre del modelo en el nombre del archivo
                route = f"Models/{dataset_id}_{nombre}_model.pkl"
                joblib.dump(model, route)
                routes.append(route)


                # Calcular métricas de rendimiento
                acc = accuracy_score(y_pred, y_test)
                pre = precision_score(y_pred, y_test, average='macro')
                rec = recall_score(y_pred, y_test, average='micro')
                f1 = f1_score(y_pred, y_test, average='weighted')

                if option_train == 2:  # Cross Validation
                    acc = cross_val_score(model, x, y, cv=5, scoring='accuracy').mean()
                    pre = cross_val_score(model, x, y, cv=5, scoring='precision_macro').mean()
                    rec = cross_val_score(model, x, y, cv=5, scoring='recall_macro').mean()
                    f1 = cross_val_score(model, x, y, cv=5, scoring='f1_macro').mean()

                #average = (acc + pre + rec + f1) / 4

                task = {
                    "accuracy": acc,
                    "precision": pre,
                    "recall": rec,
                    "f1": f1,
                    "route": route,
                    "normalization": normalization,
                    "option_train": option_train,
                    "dataset_id": dataset_id,
                    #"average": average,
                    "algorithm_id": algorithm_id,
                    "modelo": nombre,
                    "y_pred": y_pred,
                }
                tasks.append(task)

                result = coleccion_modelos.insert_one(task)
                id_insertado = str(result.inserted_id)
                


                # Modificar la variable 'route' después de insertar en la base de datos
                route_con_id = f"Models/{dataset_id}_{nombre}_model_{id_insertado}.pkl"
                os.rename(route, route_con_id)
                routes[-1] = route_con_id  # Actualizar la última ruta en la lista
                #task['route'] = route_con_id  # Actualizar la ruta en el diccionario


                # Actualizar el documento en la colección con la nueva ruta que incluye el ID
                coleccion_modelos.update_one(
                    {"_id": result.inserted_id},
                    {"$set": {"route": route_con_id}}
                )
                task['route'] = route_con_id  # Actualizar la ruta en el diccionario

            result = coleccion_entrenamientos.insert_one({'dataset_id': dataset_id, 'datosEntrenamiento': tasks})
            id_insertado_entrenamiento = str(result.inserted_id)
                

            return jsonify({'message': 'Entrenamiento exitoso.', 'routes': routes, 'idEntrenamiento': id_insertado_entrenamiento}), 200

        return jsonify({'error': 'No se ha cargado ningún archivo.'}), 400

    except Exception as e:
        print(e)
        return jsonify({'error': 'Ha ocurrido un error.'}), 400
    
@app.route('/results/<train_id>/', methods=['GET'])
def get_model_metrics(train_id):
    try:
        # Convertir el ID del registro a ObjectId de MongoDB
        object_id = ObjectId(train_id)

        # Buscar el registro por ID
        registro = coleccion_entrenamientos.find_one({'_id': object_id})

        if registro:
            # Buscar modelos entrenados para el conjunto de datos específico
            modelos_entrenados = registro['datosEntrenamiento']

            # Preparar la respuesta JSON con las métricas de cada modelo
            metrics_list = []
            for modelo_entrenado in modelos_entrenados:
                # Calcular la matriz de confusión
                # Cargar el modelo desde el archivo guardado
                
                #loaded_model = keras.models.load_model(modelo_entrenado['route'])
                print(modelo_entrenado['route'])
                # loaded_model = joblib.load(modelo_entrenado['route'])
                # Predecir en datos de prueba
                # loaded_y_pred = y_pred
                # print(loaded_y_pred)
                # loaded_y_pred_classes = np.argmax(loaded_y_pred, axis=1)
                # loaded_y_true_classes = np.argmax(y_test, axis=0)

                # # Calcular métricas por separado
                # accuracy = accuracy_score(loaded_y_true_classes, loaded_y_pred_classes)
                # precision = precision_score(loaded_y_true_classes, loaded_y_pred_classes, average='weighted')
                # recall = recall_score(loaded_y_true_classes, loaded_y_pred_classes, average='weighted')
                # f1 = f1_score(loaded_y_true_classes, loaded_y_pred_classes, average='weighted')

                # print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

                # # Obtener la matriz de confusión
                # conf_mat = confusion_matrix(loaded_y_true_classes, loaded_y_pred_classes)

                # print(conf_mat)
                loaded_y_pred = y_pred
                print(loaded_y_pred)

                loaded_y_pred_classes = np.round(loaded_y_pred).astype(int)
                loaded_y_true_classes = np.round(y_test).astype(int)

                # Calcular métricas
                accuracy = accuracy_score(loaded_y_true_classes, loaded_y_pred_classes)
                precision = precision_score(loaded_y_true_classes, loaded_y_pred_classes, average='weighted')
                recall = recall_score(loaded_y_true_classes, loaded_y_pred_classes, average='weighted')
                f1 = f1_score(loaded_y_true_classes, loaded_y_pred_classes, average='weighted')

                print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

                # Obtener la matriz de confusión
                conf_mat = confusion_matrix(loaded_y_true_classes, loaded_y_pred_classes)

                print(conf_mat)


                metrics = {
                    'accuracy': modelo_entrenado['accuracy'],
                    'precision': modelo_entrenado['precision'],
                    'recall': modelo_entrenado['recall'],
                    'f1 score': modelo_entrenado['f1'],
                    'confusion_matrix': str(conf_mat),  # Convertir la matriz a una lista para JSON
                    'Modelo': modelo_entrenado['route'],
                }
                metrics_list.append(metrics)

            return jsonify({'metrics': metrics_list}), 200
        else: 
            return jsonify({'error': 'Registro no encontrado.'}), 404

    except Exception as e:
        print(e)
        return jsonify({'error': 'Ha ocurrido un error al obtener las métricas de los modelos.'}), 500
# def results(train_id):
#     try:
#         # Convertir el ID del registro a ObjectId de MongoDB
#         # object_id = ObjectId(train_id)

#         # Buscar el registro por ID
#         registro = coleccion_entrenamientos.find_one({'_id': train_id})

#         if registro:
#             # Obtener el contenido del archivo en formato JSON
#             json_content = registro['datosEntrenamiento']

#             # Convertir el JSON a un DataFrame de Pandas
#             df = pd.DataFrame(json_content)

#             return jsonify({'message': 'Entrenamiento exitoso.', 'datosEntrenamiento': df}), 200

#         return jsonify({'error': 'No se ha cargado ningún archivo.'}), 400

#     except Exception as e:
#         print(e)
#         return jsonify({'error': 'Ha ocurrido un error.'}), 400
    
@app.route('/graficar', methods=['POST'])
def graficar():
    try:
        df_numeric = df.select_dtypes(include='number')
        plt.rcParams['figure.figsize'] = (16, 9)
        plt.style.use('ggplot')
        df_numeric.hist()
        filenameH = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + 'histograma.png'
        plt.savefig('Imagenes/histogramas/' + filenameH)

        colormap = plt.cm.coolwarm
        plt.figure(figsize=(12,12))
        plt.title('Chronic_Kidney_Disease Data Set', y=1.05, size=15)
        sb.heatmap(df_numeric.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
        filenameC = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + 'correlacion.png'
        plt.savefig('Imagenes/correlacion/' + filenameC)
        return jsonify({'histograma': 'Imagenes/histogramas/' + filenameH,
                        'correlacion':'Imagenes/correlacion/' + filenameC}), 200
    except AttributeError :
        return jsonify({'error': 'No se ha cargado ningún archivo.'}), 400
    except Exception:
        return jsonify({'error': 'Ha ocurrido un error.'}), 400

# @app.route('/entrenar', methods=['POST'])
# def train():
#     try:
#         body = request.get_json()
#         x = df[body['x']]
#         y = df[body['y']]
#         normalizacion = body['normalizacion']
#         tecnica = body['tecnica']
#         numero = body['numero']
#         X_train, X_test, y_train, y_test = [], [], [], []
#         if tecnica == 'hold':
#             X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = numero/100, random_state = 101)
#         elif tecnica == 'cross':
#             X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 1/numero, random_state = 101)

#         if normalizacion == 'standard':
#             sc_X = StandardScaler()
#             X_train = sc_X.fit_transform(X_train)
#             X_test = sc_X.transform(X_test)
#         elif normalizacion == 'minmax':
#             escalar=MinMaxScaler()
#             X_train=escalar.fit_transform(X_train)
#             X_test=escalar.transform(X_test)

#         modelo = ControladorModelo().entrenar(body['modelo'])
#         modelo.fit(X_train, y_train)
#         y_pred = modelo.predict(X_test)

#         ruta = "Models/"+ filename.split(".")[0]+"_" + body['modelo']+".pkl"
#         joblib.dump(modelo, ruta)
#         acc = accuracy_score(y_pred, y_test)
#         pre = precision_score(y_pred, y_test, average='macro')
#         rec = recall_score(y_pred, y_test, average='micro')
#         f1 = f1_score(y_pred, y_test, average='weighted')
#         if tecnica == 'cross':
#             acc = cross_val_score(modelo, x, y, cv=numero,scoring='accuracy') 
#             acc = acc.mean()
#             pre = cross_val_score(modelo, x, y, cv=numero,scoring='precision_macro')
#             pre = pre.mean()
#             rec = cross_val_score(modelo, x, y, cv=numero,scoring='recall_macro')
#             rec = rec.mean()
#             f1 = cross_val_score(modelo, x, y, cv=numero,scoring='f1_macro')
#             f1 = f1.mean()
#         promedio = (acc + pre + rec + f1)/4
#         print(acc, pre, rec, f1)
#         task = {"accuracy": acc, 
#                 "precision": pre, 
#                 "recall": rec,
#                 "f1": f1,
#                 "ruta": ruta, 
#                 "x": body['x'], 
#                 "y": body['y'], 
#                 "normalizacion": normalizacion, 
#                 "tecnica": tecnica,
#                 "numero": numero, 
#                 "nombre": filename,
#                 "promedio": promedio,
#                 "modelo": body['modelo'],
#                 }
#         myMoldel.insert_one(task)
        
#         return jsonify({'message': 'Entrenamiento exitoso.', 'nombre' : ruta}), 200
#     except AttributeError :
#         return jsonify({'error': 'No se ha cargado ningún archivo.'}), 400
#     except Exception as e:
#         print(e)
#         return jsonify({'error': 'Ha ocurrido un error.'}), 400
    

    
def convertir_a_cadena(documento):
    documento['_id'] = str(documento['_id'])
    return documento

@app.route('/listar', methods=['POST'])
def listar():
    body = request.get_json()
    nombre = body['nombre']
    
    resultados = coleccion_modelos.find({ 'nombre': nombre })
    lista_resultados = []
    for documento in resultados:
        documento = convertir_a_cadena(documento)
        lista_resultados.append(documento['modelo'])

    json_resultados = json.dumps(lista_resultados)

    return  json_resultados, 200

@app.route('/metricas', methods=['POST'])
def metricas():
    body = request.get_json()
    nombre = body['nombre']
    
    resultados = coleccion_modelos.find({ 'nombre': nombre })
    lista_resultados = []
    for documento in resultados:
        documento = convertir_a_cadena(documento)
        lista_resultados.append(documento)

    json_resultados = json.dumps(lista_resultados)

    return  json_resultados, 200

@app.route('/mejores', methods=['POST'])
def mejores():
    body = request.get_json()
    nombre = body['nombre']
    
    resultados = coleccion_modelos.find({ 'nombre': nombre }).sort('promedio', -1).limit(3)
    lista_resultados = {}
    i=1
    for documento in resultados:
        documento = convertir_a_cadena(documento)
        
        lista_resultados['TOP'+str(i)] = documento
        i+=1
        
    json_resultados = json.dumps(lista_resultados)

    return  json_resultados, 200

@app.route('/predecir', methods=['POST'])
def predict():
    try:
        body = request.get_json() 
        modelo = body['modelo']
        documento= body['documento']
        prediccion = body['prediccion']
        clf_rf = joblib.load('Models/'+modelo+'.pkl')

        doc = coleccion_codigos.find({ 'documento': documento })
        aux={}
        for documento in doc:
            documento = convertir_a_cadena(documento)
            aux=documento['codigos']
        datos=[]
        y=[]
        # se decodifican los datos
        for titulo in prediccion.keys(): 
            try:
                valor=prediccion[titulo]
                arr=aux[titulo]
                for auxdicc in arr:
                        if auxdicc['valorOg']==valor:
                            datos.append(auxdicc['valorCod'])
                            break
            except KeyError:
                datos.append(prediccion[titulo]) # para los datos numericos
        # se encuentra la columna objetivo
        for titulo in aux.keys():
            try:
                valor=prediccion[titulo]
                arr=aux[titulo]
            except:
                y=aux[titulo]


        resultado_prediccion = clf_rf.predict([datos])
        # se decodifica el resultado
        for res in y:
            if res['valorCod']==resultado_prediccion.tolist()[0]:
                resultado_prediccion=res['valorOg']
                break

        return jsonify({'prediction': resultado_prediccion}), 200
    except ValueError as e :
        print(e)
        return jsonify({'error': 'Valor no encontrado.'}), 400     

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0', port=5000) #-> para EC2
    #app.run(debug=False,host='0.0.0.0', port=9000)
