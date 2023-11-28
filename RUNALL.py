# Databricks notebook source
# MAGIC %run "./util/notebook-config"

# COMMAND ----------

dbutils.notebook.run('00_Intro', timeout_seconds=0)
dbutils.notebook.run('01_Build_PDF_Document_Index', timeout_seconds=0)
dbutils.notebook.run('02_Assemble_Application', timeout_seconds=0)
dbutils.notebook.run('03_Evaluation', timeout_seconds=0)
dbutils.notebook.run('04_Deploy_Application', timeout_seconds=0)