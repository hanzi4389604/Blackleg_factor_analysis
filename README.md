# Blackleg_factor_analysis
This project is mainly focus on buiding ML and DL models in predicting blackleg disease pendamic based on weather and farming conditions such as crop rotation and pest control

As the dataset is fairly small, data augmentation and k-fold cross validation techniques were used in this analysis. Data were augmented up to 2000 entries with augmentation, and the data were divided into 3, 4, and 5 folds respectively in cross validation (the original data set too small to be divided into higher number of fold, as the test dataset will be too small to show a consistant result).

There are three sets of factors. The 1st set is weather dataset which contain humidity, wind speed, rain fall, min temperature, and max temperature. The 2nd dataset is pests, including flea beetle, maggot root incidence, and maggot root severity. The 3rd set of factor include crop rotation history of 2017,2018,2019 and 2020. In the analysis, different sets of factor, except weather data, were deleted to observed the impacts to the final model performance. Thus, the analysis was conducted based on four types of datasets and they are 'full','no_pest','no_rotation', and 'weather_oinly' 

This project mainly exploited NLP models to process the data. The models include CNN, LSTM, Transformer, C-RNN, TextCNN, LSTM-Attention, and LSTM-adversial. The results were visulized and can be found in the folder of /weights  

# Environment:

python: 3.7.4

Cuda: 10.2 

PyTorch: 1.10.1

# Examples results. 
The results were produced by CNN model with full dataset. This will demostrate how results would be like with each of the models and datasets. 

Results in 3D:

![Alt text](https://github.com/hanzi4389604/Blackleg_factor_analysis/blob/main/3D1%20(2).png)
