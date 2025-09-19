class SelectK_Best:
     
    #Define the SelectKbest algorithm.
    @staticmethod
    def selectkbest(Indep_X,Dep_Y,n):
        from sklearn.feature_selection import SelectKBest, chi2
        test=SelectKBest(score_func=chi2,k=n)
        fit1=test.fit(Indep_X,Dep_Y)
        #summarize scores
        selectk_feature=fit1.transform(Indep_X)
        return selectk_feature
    
    #Function for split the train and test
    def split_scaler(self,Indep_X,Dep_Y):
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test=train_test_split(Indep_X, Dep_Y, test_size=0.30, random_state = 0)
        from sklearn.preprocessing import StandardScaler
        sc=StandardScaler()
        x_train=sc.fit_transform(x_train)
        x_test=sc.transform(x_test)
        return x_train,x_test,y_train,y_test
    
    #R2_value
    def r2_prediction(self,regressor,x_test,y_test):
        y_pred=regressor.predict(x_test)
        from sklearn.metrics import r2_score
        r2=r2_score(y_test,y_pred)
        return r2
    
    #ML Algorithms 
    def Linear(self,x_train,y_train,x_test,y_test):
        # Fitting K-NN to the Training set
        from sklearn.linear_model import LinearRegression
        regressor=LinearRegression()
        regressor.fit(x_train,y_train)
        r2=self.r2_prediction(regressor,x_test,y_test)
        return r2
    
    def svm_linear(self,x_train,y_train,x_test,y_test):
        from sklearn.svm import SVR
        regressor=SVR(kernel='linear')
        regressor.fit(x_train,y_train)
        r2=self.r2_prediction(regressor,x_test,y_test)
        return r2
    
    def svm_NL(self,x_train,y_train,x_test,y_test):    
            from sklearn.svm import SVR
            regressor = SVR(kernel = 'rbf')
            regressor.fit(X_train, y_train)
            r2=self.r2_prediction(regressor,X_test,y_test)
            return  r2  
         
    def Decision(self,x_train,y_train,x_test,y_test):    
            # Fitting K-NN to the Training setC
            from sklearn.tree import DecisionTreeRegressor
            regressor = DecisionTreeRegressor(random_state = 0)
            regressor.fit(X_train, y_train)
            r2=self.r2_prediction(regressor,X_test,y_test)
            return  r2  
         
    def random(self,x_train,y_train,x_test,y_test):       
            # Fitting K-NN to the Training set
            from sklearn.ensemble import RandomForestRegressor
            regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
            regressor.fit(X_train, y_train)
            r2=self.r2_prediction(regressor,X_test,y_test)
            return  r2 
    
    #load the values in dataframe
    def selectk_regression(self,acclin,accsvml,accsvmnl,accde,accre):
        import pandas as pd
        dataframe=pd.DataFrame(index=["ChiScore"],columns=["Linear",'SVMl','SVMnl','Decision','Random'])
    
        for number,index in enumerate(dataframe.index):
            dataframe.loc[index,'Linear']=acclin[number]
            dataframe.loc[index,'SVMl']=accsvml[number]
            dataframe.loc[index,'SVMnl']=accsvmnl[number]
            dataframe.loc[index,'Decision']=accde[number]
            dataframe.loc[index,'Random']=accre[number]
        return dataframe
    
    
            