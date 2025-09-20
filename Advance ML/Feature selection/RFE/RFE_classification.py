class RFE_Classification:
#using self so we can acccess the variables out side the funcion

    def rfe_features(self,Indep_x,Depe_y,n):
        from sklearn.linear_model import LogisticRegression
        log_model = LogisticRegression(solver='liblinear',max_iter=1000)

        from sklearn.ensemble import RandomForestClassifier
        RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        
        from sklearn.naive_bayes import GaussianNB
        #NB = GaussianNB()
        #NB and KNN models are not fit for that dataset so make it makedown
        from sklearn.tree import DecisionTreeClassifier
        DT= DecisionTreeClassifier(criterion = 'gini', max_features='sqrt',splitter='best',random_state = 0)

        from sklearn.svm import SVC
        svc_model = SVC(kernel = 'linear', random_state = 0)

        from sklearn.neighbors import KNeighborsClassifier 
        #knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

        rfelist=[]
        
        rfemodellist=[log_model,svc_model,RF,DT]
        from sklearn.feature_selection import RFE
        for i in rfemodellist:
            log_rfe=RFE(estimator=i, n_features_to_select=n)
            log_fit=log_rfe.fit(Indep_x,Depe_y)
            log_rfe_feature=log_fit.transform(Indep_x)
            rfelist.append(log_rfe_feature)
            Columns=log_fit.get_feature_names_out()
        return rfelist,Columns


        
    #Input and output split with StandardScaler 
    def split_scaler(self,Indep_x,Depe_y):
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test=train_test_split(Indep_x,Depe_y, test_size=0.30,random_state=0)
        from sklearn.preprocessing import StandardScaler
        sc=StandardScaler()
        x_train=sc.fit_transform(x_train)
        x_test=sc.transform(x_test)
        return x_train,x_test,y_train,y_test

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
    
    #ML Algorithms 
    def Logistic(self,x_train,y_train,x_test,y_test):
        from sklearn.linear_model import LogisticRegression
        classifier=LogisticRegression(random_state = 0)
        classifier.fit(x_train,y_train)
        classifier,Accuracy,report,x_test,y_test,cm=self.cm_prediction(classifier,x_test,y_test)
        return classifier,Accuracy,report,x_test,y_test,cm
    
    def svc_linear(self,x_train,y_train,x_test,y_test):
        from sklearn.svm import SVC
        classifier=SVC(kernel='linear')
        classifier.fit(x_train,y_train)
        classifier,Accuracy,report,x_test,y_test,cm=self.cm_prediction(classifier,x_test,y_test)
        return classifier,Accuracy,report,x_test,y_test,cm
    
    def svm_NL(self,x_train,y_train,x_test,y_test):    
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'rbf')
        classifier.fit(x_train, y_train)
        classifier,Accuracy,report,x_test,y_test,cm=self.cm_prediction(classifier,x_test,y_test)
        return classifier,Accuracy,report,x_test,y_test,cm 

    def Navie(self,x_train,y_train,x_test,y_test):
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(x_train, y_train)
        classifier,Accuracy,report,x_test,y_test,cm=self.cm_prediction(classifier,x_test,y_test)
        return  classifier,Accuracy,report,x_test,y_test,cm         
    
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
        # Fitting K-NN to the Training set
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(x_train, y_train)
        classifier,Accuracy,report,x_test,y_test,cm=self.cm_prediction(classifier,x_test,y_test)
        return classifier,Accuracy,report,x_test,y_test,cm 

    #create the data frame for loading the RFE score for each algorithms
    def rfe_classification(self,acclog,accsvc,accdes,accrf):
        import pandas as pd
        rfedataframe=pd.DataFrame(index=["Linear","SVC","DecisionTree","Random"],columns=['Linear','SVC','DecisionTree','Random'])

        for number,index in enumerate(rfedataframe.index):
            rfedataframe.loc[index,"Linear"]=acclog[number]
            rfedataframe.loc[index,"SVC"]=accsvc[number]
            rfedataframe.loc[index,"DecisionTree"]=accdes[number]
            rfedataframe.loc[index,"Random"]=accrf[number]
            #this models are not fit for the data set so cancel it in code
            
            #rfedataframe['SVMnl'][idex]=accsvmnl[number]
            #rfedataframe['KNN'][idex]=accknn[number]
            #rfedataframe['Navie'][idex]=accnav[number]
            
        return rfedataframe
            
    