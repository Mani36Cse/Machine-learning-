#it's used for Unsupervised learning
#PCA is unsupervised — it finds directions of maximum variance without any label information.

class pca:

    #Function for split the train and test
    def split_scaler(self,Indep_X,Dep_Y):
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test=train_test_split(Indep_X, Dep_Y, test_size=0.30, random_state = 0)
        from sklearn.preprocessing import StandardScaler
        sc=StandardScaler()
        x_train=sc.fit_transform(x_train)
        x_test=sc.transform(x_test)
        return x_train,x_test,y_train,y_test

    #PCA function
    def PCA_input(self,x_train,x_test,n):
        from sklearn.decomposition import PCA
        pca=PCA(n_components=n)
        x_train=pca.fit_transform(x_train)
        x_test=pca.transform(x_test)
        #The explained_variance_ratio_ tells you how much information (variance) is captured by each component.
        explained_variance = pca.explained_variance_ratio_
        return x_train,x_test,explained_variance


    #confusion_matrix
    def cm_prediction(self,classifier,x_test,y_test):
        y_pred=classifier.predict(x_test)
        from sklearn.metrics import confusion_matrix
        cm=confusion_matrix(y_test,y_pred)
        #ROC score and CM report
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import classification_report

        Accuracy=accuracy_score(y_test,y_pred)
        report=classification_report(y_test,y_pred)
        return classifier,Accuracy,report,x_test,y_test,cm

    #ML Algorithms after the PCA will get the linear line and extrate the data using that data we can perfor the Supervise --- Classification 
    def Logistic(self,x_train,y_train,x_test,y_test):
        from sklearn.linear_model import LogisticRegression
        classifier=LogisticRegression(random_state = 0)
        classifier.fit(x_train,y_train)
        classifier,Accuracy,report,x_test,y_test,cm=self.cm_prediction(classifier,x_test,y_test)
        return classifier,Accuracy,report,x_test,y_test,cm
    
    def svm_linear(self,x_train,y_train,x_test,y_test):
        from sklearn.svm import SVC
        classifier=SVC(kernel='linear')
        classifier.fit(x_train,y_train)
        classifier,Accuracy,report,x_test,y_test,cm=self.cm_prediction(classifier,x_test,y_test)
        return classifier,Accuracy,report,x_test,y_test,cm

    def knn(self,x_train,y_train,x_test,y_test):
        # Fitting K-NN to the Training set
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(x_train, y_train)
        classifier,Accuracy,report,x_test,y_test,cm=self.cm_prediction(classifier,x_test,y_test)
        return  classifier,Accuracy,report,x_test,y_test,cm
             
    def Decision(self,x_train,y_train,x_test,y_test):    
        # Fitting K-NN to the Training setC
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)
        classifier.fit(x_train, y_train)
        classifier,Accuracy,report,x_test,y_test,cm=self.cm_prediction(classifier,x_test,y_test)
        return classifier,Accuracy,report,x_test,y_test,cm  
         
    def random(self,x_train,y_train,x_test,y_test):
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(x_train, y_train)
        classifier,Accuracy,report,x_test,y_test,cm=self.cm_prediction(classifier,x_test,y_test)
        return classifier,Accuracy,report,x_test,y_test,cm

    #load the values in dataframe
    def PCA_Classification(self,acclog,accsvml,accknn,accdes,accrf):
        import pandas as pd
        dataframe=pd.DataFrame(index=["PCA"],columns=["Logistic",'SVMl','KNN','Decision','Random'])
    
        for number,index in enumerate(dataframe.index):
            dataframe.loc[index,'Logistic']=acclog[number]
            dataframe.loc[index,'SVMl']=accsvml[number]
            dataframe.loc[index,'KNN']=accknn[number]
            dataframe.loc[index,'Decision']=accdes[number]
            dataframe.loc[index,'Random']=accrf[number]
        return dataframe