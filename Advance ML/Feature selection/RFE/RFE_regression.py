class RFE_regression:
#using self so we can acccess the variables out side the funcion

    #Input the ML algorithm for select the RFE score each algorithm, N=no of columns 
    def rfe_feature(self,Indep_x,Depe_y,n):
        rfe_list=[]
        from sklearn.linear_model import LinearRegression
        lin = LinearRegression()
        
        from sklearn.svm import SVR
        SVRl = SVR(kernel = 'linear')

        #In 
        from sklearn.svm import SVR
        #SVR('rbf') taking many time to process model
        #SVRnl = SVR(kernel = 'rbf')
        
        from sklearn.tree import DecisionTreeRegressor
        dec = DecisionTreeRegressor(random_state = 0)
        
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators = 10, random_state = 0)

        rfemodellist=[lin,SVRl,dec,rf]
        from sklearn.feature_selection import RFE
        for i in rfemodellist:
            print(i)
            #select the algorithm with n in RFE
            log_rfe= RFE(estimator=i, n_features_to_select=n)
            log_fit=log_rfe.fit(Indep_x,Depe_y)
            log_rfe_feature=log_fit.transform(Indep_x)
            rfe_list.append(log_rfe_feature)
            #It will show the input columns which is select  
            Columns=log_fit.get_feature_names_out()
        return rfe_list,Columns
    
    #Input and output split with StandardScaler 
    def split_scaler(self,Indep_x,Depe_y):
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test=train_test_split(Indep_x,Depe_y, test_size=0.30,random_state=0)
        from sklearn.preprocessing import StandardScaler
        sc=StandardScaler()
        x_train=sc.fit_transform(x_train)
        x_test=sc.transform(x_test)
        return x_train,x_test,y_train,y_test

    #R2_score for evaluation Regression
    def r2_prediction(self,regression,x_test,y_test):
        y_pred=regression.predict(x_test)
        from sklearn.metrics import r2_score
        r2=r2_score(y_test,y_pred)
        return r2

    #ML regression algorithms 
    def Linear(self,x_train,y_train,x_test,y_test):
        from sklearn.linear_model import LinearRegression
        regression=LinearRegression()
        regression.fit(x_train,y_train)
        r2=self.r2_prediction(regression,x_test,y_test)
        return r2

    def svm_linear(self,x_train,y_train,x_test,y_test):
        from sklearn.svm import SVR
        regression=SVR(kernel='linear')
        regression.fit(x_train,y_train)
        r2=self.r2_prediction(regression,x_test,y_test)
        return r2
    
    def svm_NL(self,x_train,y_train,x_test,y_test):    
        from sklearn.svm import SVR
        regression = SVR(kernel = 'rbf')
        regression.fit(x_train, y_train)
        r2=self.r2_prediction(regression,x_test,y_test)
        return  r2  
         
    def Decision(self,x_train,y_train,x_test,y_test):    
        # Fitting K-NN to the Training setC
        from sklearn.tree import DecisionTreeRegressor
        regression = DecisionTreeRegressor(random_state = 0)
        regression.fit(x_train, y_train)
        r2=self.r2_prediction(regression,x_test,y_test)
        return  r2  
         
    def random(self,x_train,y_train,x_test,y_test):       
        # Fitting K-NN to the Training set
        from sklearn.ensemble import RandomForestRegressor
        regression = RandomForestRegressor(n_estimators = 10, random_state = 0)
        regression.fit(x_train, y_train)
        r2=self.r2_prediction(regression,x_test,y_test)
        return  r2 

    #create the data frame for loading the RFE score for each algorithms
    def rfe_regression(self,acclin,accsvrl,accdec,accrf):
        import pandas as pd
        rfedataframe=pd.DataFrame(index=["Linear","SVRl","DecisionTree","Random"],columns=['Linear','SVRl','DecisionTree','Random'])

        for number,idex in enumerate(rfedataframe.index):
            rfedataframe.loc[idex,"Linear"]=acclin[number]
            rfedataframe.loc[idex,"SVRl"]=accsvrl[number]
            #rfedataframe.loc[idex,"SVRnl"]=accsvrnl[number]
            rfedataframe.loc[idex,"DecisionTree"]=accrf[number]
            rfedataframe.loc[idex,"Random"]=accdec[number]
        return rfedataframe
            
    