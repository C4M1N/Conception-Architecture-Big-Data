#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import io
import time
import pyspark

import numpy as np
import pandas as pd
import tensorflow as tf
import pyspark.sql.functions as F

from PIL import Image
from typing import Iterator, Tuple
from IPython.display import display

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

from pyspark import SparkConf, SparkContext
from pyspark import keyword_only

from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, reverse, input_file_name, udf, pandas_udf, PandasUDFType
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType

from pyspark.ml.feature import StringIndexer
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml import Pipeline


# In[2]:


# %%info


# In[3]:


# simple vérification par rapport au produit AWS EMR
print(spark.conf.get("spark.driver.memory"))
print(spark.conf.get("spark.executor.memory"))


# In[2]:


# vérification des packages et des versions présentes
sc.list_packages()


# In[3]:


def gen_train_test_df(files_folder_url):
    """
    Fonction qui permet de générer un spark dataframe pour l'entrainement ou le test avec les données nécessaires.
    
    Args:
        files_folder_url (string): Adresse Amazon S3 du dossier d'images à traiter.
        
    Returns:
        df_output: le spark dataframe pour l'entraînement/test.
    """
    
    df_output = spark.read.format("binaryFile").option("pathGlobFilter", "*.jpg").option("recursiveFileLookup", "true").option("dropInvalid", True).load(files_folder_url)
    
    # création d'une colonne contenant juste le label au format string (sql.functions)
    df_output = df_output.withColumn("label", reverse(split(df_output.path, '/')).getItem(1))
    
    # sélection des colonnes essentielles
    df_output.select("path", "content", "label").show()
    
    print("Taille du spark dataframe:", df_output.count())
    print(df_output.printSchema())
    
    return df_output


# In[4]:


def display_img(spark, spark_df, img_cat, img_filename=None):
    """
    Fonction qui permet d'afficher une image de fruit d'une catégorie (la première image),
    ou grâce à sa catégorie et son nom de fichier (le premier de ce nom).
    
    Args:
        spark (SparkSession): Session Spark de connexion au cluster.
        spark_df (Dataframe): Spark DataFrame de Training ou de Testing.
        img_cat (string): Label de la catégorie du fruit.
        img_filename (string): Nom du fichier image recherché dans cette catégorie.
        
    Returns:
        display(img): Affiche l'image recherchée.
    """
    
    # enregistre la table temporaire dans le catalogue pour l'utilisation de SQL dans cette session spark
    spark_df.createOrReplaceTempView("spark_df")
    # print(spark_sess.catalog.listTables())

    if img_filename is None:
        bytes_img = spark.sql("SELECT content FROM spark_df WHERE label = '" + img_cat + "'").take(1)[0][0]
    else:
        bytes_img = spark.sql("SELECT content FROM spark_df WHERE label = '" + img_cat + "' AND path LIKE '%" + img_filename + "'").take(1)[0][0]

    print("shape of bytes_img:", np.shape(bytes_img))
    img = Image.open(skimage.io.BytesIO(bytes_img)).resize([100, 100])
    print("PIL object: ", img)

    return display(img)


# In[5]:


def model_for_feats():
    """
    Fonction qui permet de charger le modèle InceptionV3 avec les poids et enlève la dernière couche.
    """
    model = InceptionV3(include_top=False, weights=None)
    model.set_weights(model_weights_saved.value)
    
    return model


# In[6]:


def preprocess_img(bytes_img):
    """
    Fonction qui permet de récupérer l'image du spark dataframe en bytes pour l'utiliser ensuite
    pour la prédiction au bon format via le modèle choisi (InceptionV3) pour la featurisation de celle-ci.
    """
    
    # On lit l'image et on resize
    img = Image.open(io.BytesIO(bytes_img)).resize([100, 100])
    # on transformer en array (img_to_array -> keras)
    img_arr = img_to_array(img)
    return preprocess_input(img_arr)


# In[7]:


def ps_featurization(model, set_of_series):
    """
    Fonction qui permet de traiter une pandas séries d'images (bytes) et de les récupérer transformées (featurized) par le modèle. (=extraction de features)
    
    Args:
        set_of_series (pandas.Series): Pandas Series d'images transformées par preprocess_img (opération de map)
    
    Returns:
        pd.Series: Pandas Series d'images "transformées" par le modèle
    
    """
    
    # pour chaque image de la pandas series on préprocesse l'image sur format classique Bytes to array
    input_ps = np.stack(set_of_series.map(preprocess_img))
    # on récupère la série d'image transformée par le modèle
    preds_ps = model.predict(input_ps)
    # on "flatten" nos images et on retransforme en pandas Series
    output_list = [pred_ps.flatten() for pred_ps in preds_ps]
    return pd.Series(output_list)


# In[8]:


@pandas_udf('array<float>')
def pandas_udf_gen(set_of_series_batch: Iterator[pd.Series]) -> Iterator[pd.Series]:

    """
    Fonction qui est un générateur (Iterator en anglais) et qui permet de traiter,
    par lot les Pandas Series d'images pour les transformer (featuriser).
    
    Note: le décorateur est essentiel pour déclarer un objet de type pandas_udf.
    
    Args:
        set_of_series_batch (pandas.Series): correspond à un batch de données,
        qui est un des pandas.Series d'images passées en argument.
        
    Yields:
        Les pandas.Series transformées par le modèle.
    """

    model = model_for_feats()
    
    for set_of_series in set_of_series_batch:
        yield ps_featurization(model, set_of_series)


# In[9]:


class CustomTransformer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    """
    Transformer custom:  pour le pré-traitement des images via une pipeline avec spark.ml
    """
    
    input_col = Param(Params._dummy(), "input_col", "input column name.", typeConverter=TypeConverters.toString)
    output_col = Param(Params._dummy(), "output_col", "output column name.", typeConverter=TypeConverters.toString)
    
    @keyword_only
    def __init__(self, input_col: str = "input", output_col: str = "output"):
        
        super(CustomTransformer, self).__init__()
        self._setDefault(input_col=None, output_col=None)
        kwargs = self._input_kwargs
        self.set_params(**kwargs)
    
    @keyword_only
    def set_params(self, input_col: str = "input", output_col: str = "output"):
        
        kwargs = self._input_kwargs
        self._set(**kwargs)
    
    def get_input_col(self):
        
        return self.getOrDefault(self.input_col)
    
    def get_output_col(self):
        
        return self.getOrDefault(self.output_col)
    
    def _transform(self, df: DataFrame):
        
        input_col = self.get_input_col()
        output_col = self.get_output_col()
        
        # to_vector = udf(lambda x: Vectors.dense(x), VectorUDT())
        # return df.withColumn(output_col, to_vector(pandas_udf_gen(input_col)))
        return df.withColumn(output_col, pandas_udf_gen(input_col))


# In[22]:


def load_preproc_data(df_foldername, input_dir):
    """
    Fonction qui permet de recharger les données d'images transformées,
    pour une réutilisation via une pipeline (ou sans) pour l'entraînement d'un modèle.
    
    Args:
        df_foldername (string): Nom du dossier (format Parquet) à charger.
        input_dir (string): Url du dossier (Amazon S3) à indiquer pour le chargement.
        
    Returns:
        df_images (Spark Dataframe): Dataframe Spark contenant 3 colonnes (url, features, label).
    """
    
    df_images = spark.read.parquet(input_dir + df_foldername)
    # Renommage de la colonne "features_vect" en "features" pour plus de clareté
    df_images = df_images.select("path", "label", "label_idx", "features_vect")
    df_images = df_images.withColumnRenamed("features_vect", "features")
    # A garder pour contrôle dans les logs ensuites
    print(df_images.show(10))
    
    return df_images


# In[11]:


# Localisation: Entrées / Sorties
bucket_name = "p8-jc-fruits"
img_path_train = f"s3a://{bucket_name}/images/train/"
img_path_test = f"s3a://{bucket_name}/images/test/"
data_process_output = f"s3a://{bucket_name}/data/"


# In[12]:


# récupération du modèle pré-entraîné avec ImageNet via Keras: InceptionV3 (Transfer Learning ici)
model_inception_v3 = InceptionV3(include_top=False)
model_weights_saved = sc.broadcast(model_inception_v3.get_weights())
# print(model_inception_v3.summary(), "\n")


# In[13]:


def main():
    
    # Assignation de la taille des batchs pour le pré-processing via le modèle pré-entraîné
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")
    
    # génération du spark dataframe avec les images d'entraînement/test
    df_train = gen_train_test_df(files_folder_url=img_path_train)
    df_test = gen_train_test_df(files_folder_url=img_path_test)
    
    # contrôle de chargement: exemple avec visualisation d'une image (catégorie + en particulier)
    # Rappel : A commenter pour la version envoyée au driver via spark-submit !
    # display_img(spark, df_train, img_cat='banana_red')
    # display_img(spark, df_train, img_cat='lychee', img_filename='127_100.jpg')
    
    # stage de pipeline -> pré-processing des images avec transformer custom
    indexer = StringIndexer(inputCol='label', outputCol='label_idx')  # utile (spark ne prend pas de string pour le ML)
    featurizer = CustomTransformer(input_col='content', output_col='features')  # df_train, df_test
    
    # Pré-processing des images via une pipeline -> réutilisable pour préparer l'entraînement d'un modèle ensuite
    pipeline = Pipeline(stages=[indexer, featurizer])
    pipeline_model = pipeline.fit(df_train)
    df_train_feats = pipeline_model.transform(df_train)
    df_test_feats = pipeline_model.transform(df_test)
    
    # Transformation de type: passage du type array(float) à vector (pour utilisation scaling ou pca)
    # N'est pas intégré à la classe CustomTransformer après échecs d'essais pour optimisation (temps de transformation)
    to_vector = udf(lambda x: Vectors.dense(x), VectorUDT())
    df_train_feats = df_train_feats.select("path", "label", "label_idx", "features", to_vector("features").alias("features_vect"))
    df_test_feats = df_test_feats.select("path", "label", "label_idx", "features", to_vector("features").alias("features_vect"))

    # Contrôle Le dtype de la colonne 'features' en sortie -> array<float> to vector
    print("Format de sortie des features (train): {}".format(dict(df_train_feats.dtypes)['features_vect']))
    print("Format de sortie des features (test): {}".format(dict(df_test_feats.dtypes)['features_vect']))
    # Rappel: ce sont des transformations (lazy) -> seront déclenchées (exécutées) lors d'une action ensuite
    
    # contrôles
    print(df_train_feats.show(25))
    print(df_test_feats.show(25))
    
    # Enregistrement des images pré-traitées (train & test) au format Parquet
    df_train_feats.write.mode("overwrite").parquet(data_process_output + 'df_train_ready')
    df_test_feats.write.mode("overwrite").parquet(data_process_output + 'df_test_ready')
        
    # pas d'utilité à les garder en mémoire ici, si elles sont rechargées ensuite (parquet)
    del df_train_feats
    del df_test_feats
    
    return
    
if __name__ == '__main__':
    main()


# In[14]:


# start_time = time.time()
# main()
# print("Preprocessing et backup en {:.2f} secondes".format(time.time() - start_time))


# In[24]:


df_train_ready = load_preproc_data(df_foldername='df_train_ready', input_dir=data_process_output)
df_test_ready = load_preproc_data(df_foldername='df_test_ready', input_dir=data_process_output)


# In[ ]:




