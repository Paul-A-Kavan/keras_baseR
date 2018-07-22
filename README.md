# keras_baseR
A few R functions to implement an already-trained keras model without loading the keras package.

## Purpose
Building neural networks is a lot of fun but complex models come with the challenge of complicated dependencies and many working parts. For smaller (simpler) nn models I wanted the ability to run keras::predict_proba() without loading up the keras package. (i.e. train a keras model on a machine with pyhton and keras but then go implement the model in an environment without python.) This turned into the desire to run models without any heavy dependencies such as an HDF5 reader package. 

Obviously, by removing the use of these critical resources the abilities of a keras model is seriously restricted but for particular cases these functions can be helpful.


## Example
```
library(keras)
source("core.R")
source("misc.R")
source("activations.R")

# create some input data with a signal for the neural network to train toward
x <- rbind(matrix(rnorm(1000,  mean=1), ncol=10, nrow=100),
           matrix(rnorm(1000, mean=-1), ncol=10, nrow=100))

# create the responses matrix
responses <- c(rep(0,100), rep(1,100))
y <- matrix(c(responses, as.numeric(!responses)), byrow = F, ncol=2)

# randomize the training data
rand_order <- sample(1:200, 200)
x <- x[rand_order,]
y <- y[rand_order,]

# set up a simple keras model
k_mod <- keras_model_sequential() %>% 
  layer_dense(units=10, input_shape=10) %>% 
  layer_dense(units=7, activation="relu") %>% 
  layer_dense(units=2, activation="softmax") %>% 
  compile(loss = "mean_squared_error",
          optimizer = optimizer_adam(),
          metrics = c("accuracy"))
k_mod %>% 
  fit(x=x, y=y, epochs=10)

c_mod <- convert_model_baseR(k_mod)

preds <- predict_proba_baseR(c_mod, mat=x)

# note the rounding errors in this simple implementation compared to the keras package
accurate_preds <- predict_proba(k_mod, x)

head(preds)
head(accurate_preds)
```
