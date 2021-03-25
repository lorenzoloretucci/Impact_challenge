
**Data Generation**: \
In this simple model we suppose that each bin has a binary signal 0 for empty and 1 for full. These reports can come from QR codes, smart bins, etc. For now se suppose that the signals are all uncorrelated: for each bin a probability of success p is drawn uniformely at random each time and from it a time series of zeros and ones is generated. Optionally
to simulate a more realistic scenario it is possible also to generate additional features, that would represent other type of data, e.g., daily estimated garbage for that bin for
that particular day, or any other information. In this case, since we don't have access to these type of data we generate features containing all ones.\

**Model training**: \
The model depends on the type of available features. If we have only binary observations than the model can be trained in an autoregressive fashion, where we want to predict the state of each bin at time t given all the past observations of that bin. This way is relatively straightforward and an illustration can be seen in the figure below:
<p align="center">
  <img width="460" height="300" src="https://github.com/lorenzoloretucci/Impact_challenge/blob/main/backend/autoregressive%20simple.PNG">
</p>
\
In the case we have features the things become more complicated. For this we need to train a model that takes in input a vector and gives in output a binary observation thus it is 
impossible to train an autoregressive model, since we cannot feed back the output to the input. For this reason we can train a model that takes a number of "step" vectors and predicts the binary state on the next "step" timestamps. An illustration is shown below:
<p align="center">
  <img width="460" height="300" src="https://github.com/lorenzoloretucci/Impact_challenge/blob/main/backend/autoregressive%20simple.PNG">
</p>
\
**Predictions**: \
After the model is trained the predictions are made by generating a test set in the same way of the train and the predictions are made by sampling from the predicted distribution, i.e. if for a bin the model predicts 0.80 (model confidence that the state of the bin is full is 80%) then the prediction for the bin is a Bernoulli random variable with parameter 0.80. This is usually done in the context of time series classification (in NLP for example) since it is not always true that the best predicted time series is obtain by picking the class with highest confidence at each step (for this reason usually approaches like our or more sofisticated like the beam search are taken).
