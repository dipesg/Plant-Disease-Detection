# Plant-Disease-Detection

### LINK to WEBAPP
  [LINK](https://plantdisease1.herokuapp.com/)
  
### DATASET LINK
  [LINK](https://github.com/dipesg/Pretrained-Emotion-Model/blob/main/dataset.rar)

### Table of Content
  * [Overview](#overview)
  * [Architechture](#architechture)
  * [Demo](#demo)
  * [Installation](#installation)
  * [Technology Stack](technologystack)
 
 ### **Overview**
- In real world, farmers face lots of devastating loss only due to they don't know which disease is affecting their crop. This project is mainly focused to solve that problem.
- Here I take images of corn, potato and tomato which is affected by the disease **Corn-Common_rust, Potato-Early_blight and Tomato-Bacterial_spot** and train it on custom **CNN** model.

### **Architechture**
![plant-arch](https://user-images.githubusercontent.com/75604769/178108755-0455a0a0-a613-4602-9670-80c760136387.png)


### :raising_hand: Project Workflow 

Our pipeline consists of three steps:
  1. An AI model which detect plant disease.
  2. An AI model which predict if the leaves has disease or not.
  3. The output predicted disease name.
  
### 🚀 Model's performance
  - **Our Custom CNN model** perform better by giving near 95% accuracy.

### **Demo**

![plant](https://user-images.githubusercontent.com/75604769/178108836-6fbc44b0-4288-4725-9eac-6e6652b2626b.gif)


### **Installation**
- **Clone the repository:**

  ```https://github.com/dipesg/Plant-Disease-Detection.git```
  
- **Create separate conda environment:**

  ```conda create -n plant python=3.6 -y```
  
- **Activate environment:**

  ```conda activate plant```
  
- **Install all the requirements:**

  ```pip install -r requirements.txt```
  
- **Run following script to run the program:**

  ```streamlit run app.py```

![](https://forthebadge.com/images/badges/made-with-python.svg)

## :warning: Technology Stack
- **Tensorflow 1.x**
- **Streamlit**
- **OpenCV**
