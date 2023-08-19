# ForestWatchAi - Website Build and Deployment Guide

Welcome to the ForestWatchAi project repository! This guide will walk you through the steps to build, deploy, and use our website. Please follow these instructions carefully.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Building the Website](#building-the-website)
- [Deployment](#deployment)

## Prerequisites

Before you begin, make sure you have the following prerequisites installed:

- **Git**: Install Git for version control.
- **Modern Web Browser**: Ensure you have any modern web browser.
- **Python 3.x**: Install Python 3.x on your machine.

Python packages to install using Python Package Manager (pip):
- fastapi
- pymongo
- uvicorn
- python-multiplier
```
pip install fastapi keras numpy Pillow pymongo requests uvicorn tensorflow python-multiplier
```

## Getting Started :-

1. Clone all the repository to your local machine from https://github.com/ForestWatchAI

## Building the Website:-

To build the ForestWatchAi website locally and preview it, follow these steps:

1. Navigate to the project directory:
```
cd path/to/ForestWatchAi/

```
2. Reload the API:
```
uvicorn main:app --reload

```
3. The built files will be located in the dist/ directory. You can now open the index.html file in your web browser to preview the      ForestWatchAi website.

## Deployment:-

 We deploy the Node.js and FastAPI files to Render. Follow these steps:

Deploying Node.js and FastAPI to Render:

1. Sign up or log in to Render.
2. Create a new service and configure it for your FastAPI application.
3. Upload your FastAPI files code to the service.
4. Deploy the service on Render. Render will provide you with a URL to access your deployed application.
