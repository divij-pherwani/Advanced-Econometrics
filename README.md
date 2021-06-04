# Advanced-Econometrics

---
title: Who survived the Titanic disaster?  Exploring data with Multinomial Logistic
  Regression
author: 'Prepared by: Divij Pherwani (430990) Andrea Furmanek (345183)'
date: "04/06/2021"
output:
  pdf_document: default
  word_document: default
  html_document:
    df_print: paged
---


```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = FALSE,message = FALSE,warning = FALSE)
```

```{r}
# Loading Libraries
library("vcd")
library("sandwich")
library("zoo")
library("lmtest")
library("MASS")
library("aod")
library("nnet")
library("Formula")
library("miscTools")
library("maxLik")
library("mlogit")
library("car")
library("survival")
library("AER")
library("stargazer")
```



# Abstract

Below paper was prepared for the needs of the final project for Advanced Econometrics at the University of Warsaw. By using ``Multinomial Logit Model`` on the data of Titanic survivors we wanted to verify whether the socio-economic status or paying higher fare impacted passenger's probability of survival. The data is originally available on the ``Kaggle`` competition website "Titanic: Machine Learning from Disaster" and contains  data for 1.309 passengers indicating whether they survive, what was their material status, what gender they were, what age they were etc. As our explanatory variables are individual specific (they do not change across alternatives) we decided to use ``Multinomial Logit Model``. In this paper, dependent variable ``Survival`` was explained with different independent variables that were selected using multinomial logit model and verified by statistical tests. The econometric model was built in ``R`` with the use of ``mlogit, survival, stargazer packages``. The end result is a final model with significant variables that best explains survival rate.

__Keywords:__ Titanic, Survival Rate, Multinomial Logit Model, Kaggle Titanic Dataset, Data pre-processing.

# 1. Introduction

The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there were not enough lifeboats for everyone onboard,resulting in the death of 1502 out of 2224 passengers and crew. While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others. Therefore this topic seems interesting for further investigation to see which variables influenced survival rate the most.

We will verify the following hypothesis using ``multinomial logit model``:


__Hypothesis 1:__


H0: Having a seat in higher class significantly increases the chance of passenger to survive.  

H1: Having a seat in higher class did not increases the chance of passenger to survive.

__Hypothesis 2:__

H0: Paying a higher fare/having a family member on board increases the chance of passenger to survive.
  
H1: Paying a higher fare/having a family member on board does not increase the chance of passenger survival.


# 2. Literature review

The Titanic disaster resulting in the sinking of the British passenger ship with the loss of 722 passengers and crew occurred in the North Atlantic on April 15, 1912. Although it has been many years since this maritime disaster took place, research on understanding what impacts individual’s survival or death has been attracting researchers' attention. It appears that this is somewhat of a common problem to work on especially that data set is publicly available. Many researchers were exploring this data with different predictive models. For example scientists from Kansas State University applied CART methodology as well as bagging and random forests that provide quite good prediction accuracy at the level of 77%^[Whitley M., “Using statistical learning to predict survival of passengers on the RMS Titanic” K-State Research Exchange, (1), 2015, pp. 32]. Using Logistic Regression also provides satisfactory results, accuracy i.e. almost of about 95%^[Kshirsagar V., Phalke N., “Titanic Survival Analysis using Logistic Regression” International Research Journal of Engineering and Technology (IRJET), (2), 2019, pp. 90] which was obtained by researchers from University of San Francisco. They concluded that the model predicted better with binary dependent variables which means the variable has a binary value as its output. Applying other methods, like random forest model predicts even better than previous models giving 93% of accuracy^[Donges N., “Predicting the Survival of Titanic Passengers” towardsdatascience.com, (3), 2018].


# 3.Dataset and preprocessing

## 3.1.Dataset

The Titanic passenger data consists of a ``training set``, a ``test set`` and a ``gender_submission set`` all are ``.csv`` files. The training set includes the response variable ``Survived`` and 11 other descriptive variables pertaining to 891 passengers. The test set does not include the response variable, but does contain the 11 other variables for 418 passengers. Additionally, gender submission includes only response variables for test set, that's why we started our data preprocessing by merging them. 


From a sample of the RMS Titanic data, we can see the various features present for each passenger on the ship: <br />
__"Survived"__: Outcome of survival (0 = No; 1 = Yes)  <br />
__"Pclass"__: Socio-economic class (1 = Upper class; 2 = Middle class; 3 = Lower class)  <br />
__"Name"__: Name of passenger  <br />
__"Sex"__: Sex of the passenger  <br />
__"Age"__: Age of the passenger (Some entries contain NaN)  <br />
__"SibSp"__: Number of siblings and spouses of the passenger aboard  <br />
__"Parch"__: Number of parents and children of the passenger aboard  <br />
__"Ticket"__: Ticket number of the passenger  <br />
__"Fare"__: Fare paid by the passenger  <br />
__"Cabin"__: Cabin number of the passenger (Some entries contain NaN)  <br />
__"Embarked"__: Port of embarkation of the passenger (C = Cherbourg; Q = Queenstown; S = Southampton)  <br />


```{r}
mosaic(~Class+Sex+Age+Survived, data=Titanic, shade=TRUE, legend=TRUE)
```

## 3.2.Preprocessing


As previously mentioned we started our data preprocessing with merging respectively datasets to get complete data for further transformations.

```{r}
# Loading Dataset
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
dfx <- read.csv("train.csv")
df1 <- read.csv("test.csv")
df2 <- read.csv("gender_submission.csv")
dfy <- merge(df1, df2)
df <- rbind(dfx, dfy)
head(df)
```


```{r}
# EDA
summary(df)
```

Since the data can have missing fields, incomplete fields, or fields containing hidden or useless information, a crucial step is to remove them in order not to complicate the further analysis. Variables ``Fare``, ``Age`` and ``Embarked`` will not be used, so we decided to remove them. Especially  embarked variable will not be used as is not individual specific. 

Our second hypothesis has to verify whether having a family on board has increased the chance of passenger to survive, therefore a new variable ``Family`` has been created as a sum of already existing variables ``SibSp`` - number of siblings and spouses of the passenger aboard and ``Parch``- number of parents and children of the passenger aboard.

```{r}
# Processing & Cleaning Data set
fdf <- subset(df, df$Embarked != "")
fdf <- subset(fdf, fdf$Fare != "")
fdf <- subset(fdf, fdf$Age != "")
fdf$Family <- fdf$SibSp + fdf$Parch
fdf$Survived <- ifelse(fdf$Survived == 0, "No", "Yes")
fdf$Pclass <- as.factor(fdf$Pclass)
fdf$Embarked <- as.factor(fdf$Embarked)
fdf$Survived <- as.factor(fdf$Survived)
fdf$Sex <- as.factor(fdf$Sex)
head(fdf)
```


# 4. Application of Econometric Models


```{r}
# Filtering Data 
fdf <- fdf[,c("Sex", "Pclass", "SibSp", "Parch", "Family", "Embarked", "Survived", "Fare", "Age")]
fdf$Farepp <- fdf$Fare/(fdf$Family + 1)
```

# Exploring and Test Multinom Models 

Generating 4 models using multinom function. 

Formulas for models, 

model1 <- Survived ~ Sex + Pclass + Age + Family + Fare + Farepp + Embarked + SibSp + Parch
model2 <- Survived ~ Sex + Pclass + Age + Family + Embarked + SibSp + Parch 
model3 <- Survived ~ Sex + Pclass + Age + Family + SibSp + Parch
model4 <- Survived ~ Sex + Pclass + Age + Family  -



```{r}
# Modeling
model1 <- multinom(Survived ~ ., data = fdf)
model2 <- multinom(Survived ~ .-Fare - Farepp, data = fdf)
model3 <- multinom(Survived ~ .-Fare - Farepp-Embarked, fdf)
model4 <- multinom(Survived ~ .-Fare - Farepp-Embarked-SibSp-Parch, fdf)
```


```{r}
#summary(model1); summary(model2); summary(model3); summary(model4)
```

Performing statistical significance

```{r}
# statistical significance
z1 <- summary(model1)$coefficients/summary(model1)$standard.errors
z2 <- summary(model2)$coefficients/summary(model2)$standard.errors
z3 <- summary(model3)$coefficients/summary(model3)$standard.errors
z4 <- summary(model4)$coefficients/summary(model4)$standard.errors
stargazer(z1,z2,z3,z4, type = "text")
```

Performing 2-tailed z test

```{r}
# 2-tailed z test
p1 <- (1 - pnorm(abs(z1), 0, 1)) * 2
p2 <- (1 - pnorm(abs(z2), 0, 1)) * 2
p3 <- (1 - pnorm(abs(z3), 0, 1)) * 2
p4 <- (1 - pnorm(abs(z4), 0, 1)) * 2
stargazer(p1,p2,p3,p4, type = "text")
```



```{r}
# Comparing all Models
stargazer(model1, model2, model3, model4, type = "text")
```

# Exploring MLogit Data and Model

Generating data in mlogit.data format

```{r}
# Generating data in mlogit.data format
mldf <- mlogit.data(fdf, shape = "wide", choice= "Survived")
head(mldf)
```

Running the model on mlogit data

```{r}
# Running the model on mlogit data
model <- mlogit(Survived~ 0| Sex + Family + Pclass + Age, data = mldf)
model
```

Summary of the mlogit model

```{r}
# Summary of the mlogit model 
summary(model)
```
# 5. Results & Findings

## Part A

We compare the restricted model to the unrestricted model to obtain the joint significance test. Number of restrictions are 4. We compare the restricted model to the constant model using likelihood ratio test.

p-value = 2.2 e-16 and LR test = 615.11

Null hypothesis: Pclass = 0, Sex = 0, Family = 0 and Age = 0

The p-value is lower than the significance level of 0.05. Therefore, we have to reject the null hypothesis which states parameters are jointly insignificant. **The parameters are jointly significant. **

```{r}
# General-to-specific method for variables selection
x <- multinom(Survived~ Pclass + Sex + Family + Age, data = fdf)
y <- multinom(Survived~1, data=fdf)
lrtest(x, y)
```

## Part B


The model is created with non linear relation (square of Fare) and interactions between Age:Pclass & Sex:Pclass. The Fare and Fare^2 both have a p-value < 0.05. Similarly, for Pclass2:Age, Sexmale:Pclass2 and Sexmale:Pclass3 have p-value < 0.05. This means, we reject the null hypothesis. The parameters add meaningful addition to model. Pclass:Age interaction is insignificant.    

```{r}
# One nonlinear relationship and interaction between variables

fdf$Fare2 <- fdf$Fare^2
fdf$Farepp2 <- fdf$Farepp^2
nlr_model <- multinom(Survived ~ Fare + Fare2 + Sex + Pclass + Age+ Age*Pclass 
                      + Sex * Pclass, data = fdf)
stargazer(nlr_model, type = "text")

```

## Part C

Marginal effects are used to quantify the incremental risk associated with each factor.We start by creating a vector z which contains average characterstics. 

```{r}
# Calculation and interpretation of marginal effects for the model 

model <- mlogit(Survived~ 0| Fare + Family +Age, data = mldf)
z <- with(mldf, data.frame(Fare = tapply(Fare, index(model)$alt, mean), Family = 
                             tapply(Family, index(model)$alt, mean), 
                           Age = tapply(Age, index(model)$alt, mean)))

```


The effect of Fare on Survival for each increment is by 0.3%. More the fare, more chances of survival. 

```{r}
# Marginal effects for Fare
x <- effects(model, covariate = "Fare", data = z)
stargazer(x, type = "text")
```

The effect of Family size on Survival for each increment in Family members is by -0.7%. More the family, lesser the chances of survival.

```{r}
# Marginal effects for Family
y <- effects(model, covariate = "Family", data = z)
stargazer(y, type = "text")

```

The effect of Age on Survival for each increment in Age is by -0.4%. Older the person, lesser the chances of survival.

```{r}
# Marginal effects for Age
z <- effects(model, covariate = "Age", data = z)
stargazer(z, type = "text")
```


## Part D

The final model is (3) in the table which can be represented by the formula:

log(P(Survived=Yes)/P(Survived=No)) = 4.411 + (-3.703 * Sexmale) + (-1.306 * Pclass2) + (-2.271 * Pclass3) + (-0.199 * Family) + (-0.036 * Age) 

```{r}
#present models in one quality table

final <- multinom(Survived~ Sex + Family + Pclass + Age + Pclass*Sex, data = fdf)

stargazer(model1, final, model4, type="text")

```


## Part E

The linktest is used to check the appropriate form of the model. The motivation behind the link test is the idea that if a regression is specified appropriately you should not be able to find additional independent variables. 

The condition for linktest is to have yhat as significant and  yhat^2 as insgnificant. (Ho: The model has appropriate form)

In the below model, we have yhat as significant and yhat2 as insignificant. Therefore, we fail to reject the null hypothesis. The model has an an appropriate form. 



```{r}
# perform the linktest and interpret the result

linktest = function(model) 
{
  y = as.numeric(fdf$Survived) -1
  yhat = log(model$fitted.values/(1-model$fitted.values))
  yhat2 = yhat^2
  # auxiliary regression
  aux.reg = multinom(y~yhat+yhat2)
  show(summary(aux.reg))
  return(aux.reg)
}

mylogit <- multinom(Survived~ Sex + Pclass + Age + Family + Sex*Pclass, data=fdf)
lt <- linktest(mylogit)
stargazer(lt, type="text")
```



## Hypothesis Testing 

**Hypothesis 1**

H0: Having a seat in higher class insignificant to survive.  

H1: Having a seat in higher class is significant to survive.



**Result 1**

The parameters Pclass2 and Pclass3 are significant as their p-value is < 0.05. Having a seat in class 2 and class 3 is significant to survival but as the coefficient is negative it decreases the probability of survival.The rate of survival is class 2 is more than in class 3. The rate of survival in class 1 is the most as pclass 2 = 0 and pclass 3 = 0.

Therefore, we reject the null hypothesis. Having a seat in higher class indeed increases the chance of passenger to survive. 


**Hypothesis 2**

H0: Paying a higher fare/having a family member on board is insignificant to survive.
  
H1: Paying a higher fare/having a family member on board is significant survival.


**Result 2**

Fare is insignificant as the p-value is greater than 0.05. We fail to reject the null hypothesis. Paying a higher fare is insignificant to survival.

However, having a family member on board is significant as the p-value is less than 0.05. We reject the null hypothesis. Having a family member on board is significant for survival. As the coefficient is negative, it is in fact reducing the probability to survive significantly. 

```{r}
# Hypothesis Testing 
hypothesis <- multinom(Survived ~ Pclass + Age + Sex + Family + Fare, data = fdf)
stargazer(hypothesis, type = "text")

```

# Conclusions

In our analysis, we concluded that an econometric model can be useful in predicting what features increases survival rate during sinking of the Titanic in 1912. As we know from literature one of the reasons that the shipwreck lead to such loss of life is that were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, like younger passengers, the upper-class and those who had no family members on board which we confirmed with the results of the Multinomial Logit Model.



# Appendix 

```{r, ref.label=knitr::all_labels(),echo=TRUE,eval=FALSE}

```
