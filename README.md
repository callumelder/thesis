# Introduction

This is the Git repository for the ARMHub bot Databricks notebook. It will process documents and prepare an endpoint
for the Teams Armbot to call.

# Instructions

Although this is not completely automated, there are a few manual steps required to get started. Note, the
{user_name} reference in these instructions denotes your ARM Hub AD username (without the @armhub.com.au),
for example: dennis_mellican.

1. Clone this repo in Databricks: Repos -> Workspace. Use a unique name, such as your username.
1. Create or rename your volume to follow this naming convention: /Catalog/Volume/{user_name}/{user_name}_volume.
1. Upload documents to the aforementioned volume.
1. Attach the notebooks to a cluster, such as your cluster name (start it now).
1. Select the RUNALL notebook and click the "Run all" button.
1. Your endpoint will be something like (replace <user_name>):
    ```https://adb-4738734208814752.12.azuredatabricks.net/serving-endpoints/llm-{user_name}-armhub-endpoint/invocations```

Remember to shutdown your cluster afterwards. This does not affect the endpoint.