## Audio Emotion Recognition Using Neural Network [AER]

# Demo
https://emotion-classifier-061q.onrender.com/

https://user-images.githubusercontent.com/90950629/228269987-2897ddbc-ea73-4725-8861-fc95cedb0a6a.mp4



# Implemented By

Purbayan Majumder,Rubina Das,Aman Khan,Jaya Gupta,Ujjwal Kumar,Sadique Akhtar

# Project Overview
  As humans were are well trained in our experience reading recognition of various emotions which make us more sensible and understandable, but can be a difficult task for computers.AER is the task of recognizing the emotional aspects of speech irrespective of the semantic contents.In this project we have built a deep learning model that is trained on emotion detection or better yet emotion classification using audio data . We have an extract feature function which is using some other functions to extract information from the audio file . We had used the MLP Classifier which stands for Multi layer    Perceptron Classifier. We then have prepared our very own websited using Django and then hosted it on AWS and heroku . 
  
# Applications

1.In the workplace --->
Speech recognition technology in the workplace has evolved into incorporating simple tasks to increase efficiency, as well as beyond tasks that have traditionally needed humans, to be performed.

2. In banking --->
The aim of the banking and financial industry is for speech recognition to reduce friction for the customer. Voice-activated banking could largely reduce the need for human customer service, and lower employee costs. A personalised banking assistant could in return boost customer satisfaction and loyalty.

3. In marketing --->
Voice-search has the potential to add a new dimension to the way marketers reach their consumers. With the change in how people are going to be interacting with their devices, marketers should look for developing trends in user data and behaviour.

4. In Healthcare --->
In an environment where seconds are crucial and sterile operating conditions are a priority, hands-free, immediate access to information can have a significantly positive impact on patient safety and medical efficiency.

5. With the Internet of Things --->
Siri’s ability to connect to smart lights and smart thermostats24 makes it seem as though instructing your digital assistant to turn the kettle on is not far off. The Internet of Things (IoT) is not the futuristic possibility it once was, but rather a relevant development happening around us.


# Real Life World Applications 

Popular digital assistants, include:

Amazon’s Alexa,
Apple’s Siri,
Google’s Google Assistant,
Microsoft’s Cortana.

# How does it Works Now ???

#  Libraries used 

  1.librosa is a python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems.

 2. SoundFile is a python package which can read and write sound files.

 3. With the help of the Python glob module, we can search for all the path names which are looking for files matching a specific pattern (which is           defined by us). 

 4. The pickle module is used for implementing binary protocols for serializing and de-serializing a Python object structure. 
 
 5.NumPy is a Python library used for working with arrays.
 
 6.Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. Matplotlib makes easy things easy and      hard things possible.
 
 7.Scikit-learn (Sklearn) is the most useful and robust library for machine learning in Python. It provides a selection of efficient tools for machine      learning and statistical modeling including classification, regression, clustering and dimensionality reduction via a consistence interface in Python.    This library, which is largely written in Python, is built upon NumPy, SciPy and Matplotlib.
 
 
 # How is the neural network working ???
 
 # Traditional Model Vs MLP Classifier 
 
 ![image](https://user-images.githubusercontent.com/90950629/160268581-c7212a4f-daef-4b2a-97e5-7ea6ef476cb0.png)

Lets Consider these are the layers of the neural network with the first one being the input layer through which we feed our data and the last layer being the output layer through which we get the results. Each layer has a weight (w) and a bias (b).In this traditional Model when we feed our data through the first layer of the network.The output of the first layer becomes the input of the second layer and the output of the second layer becomes the input of the third layer and so on. Lets see we are trying out a classification prediction based on 0 and 1 and we get our results from the last layer that y = 0,but our actual training set was 0.Then the model thinks that it had predicted wrong then we jump through the hidden layers in backward direction upto the first input layer but in between we also keep updating the bias and weights of each layers i:e Backward Propagation.The model does this until it gets the correct prediction based on the training set. But it MLP Classifier we feedforward activations to the output layer through a filtration of weighting matrices, compute a cost function that describes how good our estimate is, and backpropagate through the hidden layers to adjust the weights using gradient descent to minimize said cost function.  

The cost function basically tells us the ‘goodness of fit’ of our model in its current state, and it is something we want to optimize. 

The purpose of backpropagation is to adjust the weights between each layer to minimize the cost function.

Feedforward neural networks are also known as Multi-layered Network of Neurons (MLN). These networks of models are called feedforward because the information only travels forward in the neural network, through the input nodes then through the hidden layers (single or many layers) and finally through the output nodes. In MLN there are no feedback connections such that the output of the network is fed back into itself. These networks are represented by a combination of many simpler models(sigmoid neurons).

This MLP Classifier is a lot more timesaving and reduces the computational costs so that the model can be implemeneted easily on lower end devices .

# Feature Extracting from the audio files

                            def extract_feature(file_name, mfcc, chroma, mel):
                                with soundfile.SoundFile(file_name) as sound_file:
                                     X = sound_file.read(dtype="float32") # reading the audio file
                                     sample_rate=sound_file.samplerate
                                          if chroma:
                                                stft=np.abs(librosa.stft(X)) # getting the [only absolute values] chroma values of an audio file in an numpy array
                                                result=np.array([])
                                          if mfcc:
                                                mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0) # 40 coefficient and the sample file
                                                result=np.hstack((result, mfccs))
                                          if chroma:
                                               chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
                                               result=np.hstack((result, chroma))
                                            # Mel are the collections of men spectogram frequencies {spectrogram is the visual representation of the frequencies of the audio against time }            
                                         if mel:
                                              mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0) 
                                              result=np.hstack((result, mel)) # return result # all the extracted information to be sent as an array


Chroma[Audio Chroma] which pertains to the 12 different pitch classes of an audio file . So what happens is that if a certain pitch or pitch class is present in the audio file, it informs us that this certain pitch class is present in the audio file.It's result's are usually in the form of 0's and 1's.
The main purpose of using audio chroma is to determine the harmonic and melodic character of an audio file by remaining robust to the changes of timber and instrumentaion.

Next we first obtain the sample rate from the file using the sample audio file 
                                    
                                          with soundfile.SoundFile(file_name) as sound_file:
                                             X = sound_file.read(dtype="float32") # reading the audio file
                                             sample_rate=sound_file.samplerate
                                             
and then through librosa STFT and then we are taking the sound file and getting the required values for the chroma back in numpy[np] array and we are taking only the absolute value and setting the values either with 0 or positive values. Here it would be only  the positive values . 

                                          if chroma:
                                                stft=np.abs(librosa.stft(X)) # getting the [only absolute values] chroma values of an audio file in an numpy array

Next is the MFCC [Most frequent considered Coefficients] which represents the short term power spectrum of the sound.So what happens is that there is time power envelope for the audio signal and it is represented by the vocal track what we are trying is get is the vocal track information from the audio 
file.

Example -->  For example in a party a person is speaking not so clearly so MFCC will take coefficients from that and forms a expression of some kind and then returns a vector form of that . 

                                              if mfcc:
                                                mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0) # 40 coefficient and the sample file

Here we have taken 40 Coefficients for mfcc and the sample rate of the file and the sample file itself and then we are taking the mean of that and inputting it into our results array and again the same thing is happening in the chroma part again .

Next thing is MEL, it is nothing but Mel spectrogram frequency, it has collection of frequencies which are regarded in the mel spectrogram.We are using the MEL spectrogram feature from the librosa package and then taking the transform of that and then returing the mean value of that in the resulting numpy array.

                                              if mel:
                                              mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0) 
                                              result=np.hstack((result, mel)) # return result # all the extracted information to be sent as an array
                                              

# Dataset 

The Dataset that has been used to train our model is The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) and it has these many emotions.The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) Dataset from Kaggle contains 1440 audio files from 24 Actors vocalizing two lexically-matched statements. 

{https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio}

                                                    emotions={
                                                       '01':'neutral',
                                                        '02':'calm',
                                                        '03':'happy',
                                                        '04':'sad',
                                                        '05':'angry',
                                                        '06':'fearful',
                                                        '07':'disgust',
                                                        '08':'surprised'
                                                      }
                                                      
But for the sake of simplicity we are only going to use 4 classes of emotion to train our dataset.

    observed_emotions=['calm', 'happy', 'fearful', 'disgust']
    
    
    
# Training and testing Data

    print((x_train.shape[0], x_test.shape[0]))
    (576, 192)
    
Now we extract the features from our training dataset 
            
            print(f'Features extracted: {x_train.shape[1]}')
            Features extracted: 180
            
Then we have initialised a multilayer perceptron classifier 
      
          model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
Here the hidden layer is in tuple values which showcases the number of neurons that are present in each of the hidden layers, and the max iterarion which is also known as the epoch is set to 500.

Score of this model --> 0.8298611111111112         
Accuracy of this model ---> Accuracy: 70.31%

Next we have plotted the loss curve for this model 
 ![image](https://user-images.githubusercontent.com/90950629/160270629-3469e66e-e611-4967-8ad7-db81cf472548.png)
 
 We can observe and find that when we have reached 35 we have a very small amount of loss.
 
 Now yet again we try to implement a another MLP Classifier model
 
    model1=MLPClassifier(alpha=0.001, batch_size=128, hidden_layer_sizes=(200, 200, 100, 50), learning_rate='adaptive', max_iter=500)
    
But here instead of 1 we have used 4 layers of 200,200,100,50 neurons respectively and reduced the batch size a bit.

Next we have plotted the loss curve.
![image](https://user-images.githubusercontent.com/90950629/160270820-9ead2950-15cc-47cf-a418-5042500737ad.png)

We see that loss was going considerably low till we reached more than 200th iteration.

Score of this model ---> 0.9791666666666666
Accuracy of this model ---> Accuracy: 73.96%

The model is overfitting a bit but since we are getting more accuracy we use this.














