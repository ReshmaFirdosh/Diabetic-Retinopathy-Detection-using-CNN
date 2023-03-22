def runtrainingcode():

    #!/usr/bin/env python
    # coding: utf-8

    # In[1]:


    # This Python 3 environment comes with many helpful analytics libraries installed
    # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
    # For example, here's several helpful packages to load

    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

    # Input data files are available in the read-only "../input/" directory
    # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

    # import os
    # for dirname, _, filenames in os.walk('/kaggle/input'):
    #     for filename in filenames:
    #         print(os.path.join(dirname, filename))

    # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
    # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


    # In[2]:


    # Necessary utility modules and libraries
    import os
    import shutil
    import pathlib
    import random
    import datetime

    # Plotting libraries 
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import seaborn as sns

    # Libraries for building the model
    import tensorflow as tf
    import tensorflow_hub as hub
    from tensorflow.keras.preprocessing import image_dataset_from_directory
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D, BatchNormalization, GlobalMaxPooling2D
    from tensorflow.keras.applications import DenseNet121, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0
    from tensorflow.keras.models import Sequential
    from tensorflow.keras import backend
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import cohen_kappa_score


    # In[3]:


    root_dir = r"C:\Users\SAi Priya\Downloads\archive (2)\gaussian_filtered_images\gaussian_filtered_images"
    classes = os.listdir(root_dir)
    classes.pop(3)
    classes


    # In[4]:


    # Walk through gaussian_filtered_images directory and list names of files
    for dirpath, dirnames, filenames in os.walk(root_dir):
        print(f"There are {len(filenames)} images in {dirpath.split('/')[-1]}")


    # In[5]:


    # View random images in the dataset
    def view_random_images(root_dir=root_dir, classes=classes):
        return
        class_paths = [root_dir + "/" + image_class for image_class in classes]
        # print(class_paths)
        images_path = []
        labels = []
        for i in range(len(class_paths)):
            random_images = random.sample(os.listdir(class_paths[i]), 10)
            random_images_path = [class_paths[i]+'/'+img for img in random_images]
            for j in random_images_path:
                images_path.append(j)
                labels.append(classes[i])
        images_path
        
        plt.figure(figsize=(17, 10))
        plt.suptitle("Image Dataset", fontsize=20)

        for i in range(1, 51):
            plt.subplot(5, 10, i)
            img = mpimg.imread(images_path[i-1])
            plt.imshow(img, aspect="auto")
            plt.title(labels[i-1])
            plt.axis(False);


    # In[6]:


    # Observing the images
    view_random_images()


    # In[7]:


    train_csv = pd.read_csv(r"C:\Users\SAi Priya\Downloads\archive (2)\train.csv")
    train_csv


    # In[8]:


    train_csv['diagnosis'].value_counts()


    # In[9]:


    train_df = {}
    test_df = {}
    for i in range(5):
        df = train_csv[train_csv['diagnosis']==i]['id_code'].to_list()
        for j in random.sample(df, int(0.8*len(df))):
            train_df[j] = i
        for j in df:
            if j not in train_df.keys():
                test_df[j] = i
    train_df = pd.DataFrame(train_df.items(), columns=['id_code', 'diagnosis']).sample(frac=1, random_state=42)
    test_df = pd.DataFrame(test_df.items(), columns=['id_code', 'diagnosis']).sample(frac=1, random_state=42)
    train_df


    # In[10]:


    def mapping(df):
        class_code = {0: "No_DR",
                    1: "Mild", 
                    2: "Moderate",
                    3: "Severe",
                    4: "Proliferate_DR"}
        df['label'] = list(map(class_code.get, df['diagnosis']))
        df['path'] = [i[1]['label']+'/'+i[1]['id_code']+'.png' for i in df.iterrows()]
        return df


    # In[11]:


    mapping(train_df), mapping(test_df)


    # In[12]:


    len(train_df), len(test_df)


    # In[13]:


    # Initializing the input size
    IMG_SHAPE = (224, 224)
    N_SPLIT = 5
    EPOCHS = 1


    # In[14]:


    # Function to perform k-fold validation on test model
    def validation_k_fold(model_test, k=5, epochs=EPOCHS, n_splits=N_SPLIT, lr=0.001): 
        kfold = StratifiedKFold(n_splits=N_SPLIT,shuffle=True,random_state=42)
        train_datagen = ImageDataGenerator(rescale = 1./255)
        validation_datagen = ImageDataGenerator(rescale = 1./255)
        test_datagen = ImageDataGenerator(rescale= 1./255)

        train_y = train_df['label']
        train_x = train_df['path']

        # Variable for keeping the count of the splits we're executing
        j = 0
        es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        for train_idx, val_idx in list(kfold.split(train_x,train_y)):
            x_train_df = train_df.iloc[train_idx]
            x_valid_df = train_df.iloc[val_idx]
            j+=1
            train_data = train_datagen.flow_from_dataframe(dataframe=x_train_df, 
                                                        directory=root_dir,
                                                        x_col='path',
                                                        y_col='label',
                                                        class_mode="categorical",
                                                        target_size=IMG_SHAPE)
            
            valid_data = validation_datagen.flow_from_dataframe(dataframe=x_valid_df, 
                                                            directory=root_dir,
                                                            x_col='path',
                                                            y_col='label',
                                                            class_mode="categorical",
                                                            target_size=IMG_SHAPE)
            
            test_data = test_datagen.flow_from_dataframe(dataframe=test_df, 
                                                    directory=root_dir,
                                                    x_col='path',
                                                    y_col='label',
                                                    class_mode="categorical",
                                                    target_size=IMG_SHAPE)
                
            # Initializing the early stopping callback
            es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
            
            # Compile the model
            model_test.compile(loss='categorical_crossentropy',
                                optimizer=tf.keras.optimizers.Adamax(learning_rate=lr),
                                metrics=['accuracy'])
            history = model_test.fit_generator(train_data,
                                            validation_data=valid_data,
                                            epochs=epochs,
                                            validation_steps=len(valid_data),
                                            callbacks=[es])
            # Evaluate the model
            result = model_test.evaluate(test_data)
            model_test_result = {
                "test_loss": result[0],
                "test_accuracy": result[1],
            }
            y_pred = model_test.predict(test_data)
            return [history, model_test_result, y_pred, test_data.classes]


    # In[15]:


    # Function to plot the performance metrics
    def plot_result(hist):
        plt.figure(figsize=(10, 7));
        plt.suptitle(f"Performance Metrics", fontsize=20)
        
        # Actual and validation losses
        plt.subplot(2, 2, 1);
        plt.plot(hist.history['loss'], label='train')
        plt.plot(hist.history['val_loss'], label='validation')
        plt.title('Train and validation loss curve')
        plt.legend()

        # Actual and validation accuracy
        plt.subplot(2, 2, 2);
        plt.plot(hist.history['accuracy'], label='train')
        plt.plot(hist.history['val_accuracy'], label='validation')
        plt.title('Training and validation accuracy curve')
        plt.legend()


    # In[16]:


    # Basic CNN model for AlexNet
    model_alexnet = tf.keras.Sequential([
        Conv2D(input_shape=IMG_SHAPE+(3,), filters=96,kernel_size=11,strides=4,activation='relu'),
        MaxPool2D(pool_size=3,strides=2),
        Conv2D(filters=256,kernel_size=5,strides=1,padding='valid',activation='relu'),
        MaxPool2D(pool_size=3,strides=2),
        Conv2D(filters=384,kernel_size=3,strides=1,padding='same',activation='relu'),
        Conv2D(filters=384,kernel_size=3,strides=1,padding='same',activation='relu'),
        Conv2D(filters=256,kernel_size=3,strides=1,padding='same',activation='relu'),
        MaxPool2D(pool_size=3,strides=2),
        Flatten(),
        Dense(len(classes), activation='softmax')
    ], name="model_AlexNet")


    # In[17]:


    # Summary of AlexNet model
    # from tensorflow.keras.utils import plot_model
    # plot_model(model_alexnet)
    model_alexnet.summary()


    # In[19]:


    model_alexnet_history, model_alexnet_result, model_alexnet_pred, y_test = validation_k_fold(model_alexnet, lr=0.001)


    # In[20]:


    # Evaluation metrics for alexnet
    model_alexnet_result


    # In[21]:


    # Performance metrics for AlexNet
    plot_result(model_alexnet_history)


    # In[22]:


    y_pred_alexnet = np.argmax(model_alexnet_pred, axis=1)
    confusion_mtx = confusion_matrix(y_pred_alexnet, y_test)
    f,ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(confusion_mtx, annot=True, 
                linewidths=0.01,
                linecolor="white", 
                fmt= '.1f',ax=ax,)
    sns.color_palette("rocket", as_cmap=True)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    patient_labels = ['4', '3', '2', '1', '0']
    ax.yaxis.set_ticklabels(patient_labels)
    plt.title("Confusion Matrix - AlexNet")
    plt.show()


    # In[23]:


    print(classification_report(y_test, y_pred_alexnet))


    # In[24]:


    class_labels = ['No DR','Mild','Moderate','Severe','Proliferate']
    y_score = label_binarize(y_pred_alexnet, classes = [0,1,2,3,4])
    ytest_binary = label_binarize(y_test, classes = [0,1,2,3,4]) # one hot encode the test data true labels
    n_classes = y_score.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict() 
    # compute fpr and tpr with roc_curve from the ytest true labels to the scores
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(ytest_binary[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # plot each class  curve on single graph for multi-class one vs all classification
    f,ax = plt.subplots(figsize=(10, 10))
    colors = ['blue', 'red', 'green', 'brown', 'purple']
    for i, color, lbl in zip(range(n_classes), colors, class_labels):
        plt.plot(fpr[i], tpr[i], color = color, lw = 1.5,
        label = 'ROC Curve of class {0} (area = {1:0.3f})'.format(lbl, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw = 1.5)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for AlexNet Diabetic Retinopathy Detection Multi-Class Data')
    plt.legend(loc = 'lower right', prop = {'size': 6})
    plt.show()


    # In[ ]:


    model_alexnet.save('model.sav')

