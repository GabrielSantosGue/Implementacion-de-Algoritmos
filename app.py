#-----NLP
import csv
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#-------flask
from flask import Flask,render_template,request,redirect,url_for
app = Flask(__name__)
#formula=0
@app.route('/')
def Index():

    return render_template('index.html')

@app.route('/add_resultadoRs')
def rstudio():
        if request.method == 'POST':
            buscar=request.form['consulta']
            numero=request.form['numero']
            numero=int(numero)
            lenguaje=request.form['lenguaje']
            frsul=[]

            
            print("aquii!")
            dt=API(buscar,lenguaje,numero)

        else:   
            
            iris = load_iris()
            x=iris.data
            y=iris.target
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
            dt=RL(X_train, X_test, y_train, y_test)
            dnb=NB(X_train, X_test, y_train, y_test)
            drf=RF(X_train, X_test, y_train, y_test)
            dknn=KNN(X_train, X_test, y_train, y_test)
            dsvm=SVM(X_train, X_test, y_train, y_test)
            dn=n(X_train, X_test, y_train, y_test)
            # buscar="coronavirus"
            # lenguaje="es"
            # numero=int("10")
            # frsul=[]
            print("aquii!2")
            #render_template('index.html',dt=dt,frsul=frsul)
            # dt=API(buscar,lenguaje,numero)
            name={"ACCURACY":"0","PRESICION":"3","RECALL":"4","F-MEASURE":"7"}
        return render_template('iris.html',dt=dt,dnb=dnb,drf=drf,dknn=dknn,dsvm=dsvm,dn=dn)


@app.route('/add_data')
def vidrio():
        if request.method == 'POST':
            buscar=request.form['consulta']
            numero=request.form['numero']
            numero=int(numero)
            lenguaje=request.form['lenguaje']
            frsul=[]

            
            print("aquii!")
            dt=API(buscar,lenguaje,numero)

        else:   
            
            ruta='glasss.csv'
            dataset = pd.read_csv(ruta,delimiter=";",encoding="utf-8")
            #print (dataset)
            x=dataset.get(['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe'])
            y=dataset.get('Tipo de vidrio')
            seed = 7
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

            dt=RL(X_train, X_test, y_train, y_test)
            dnb=NB(X_train, X_test, y_train, y_test)
            drf=RF(X_train, X_test, y_train, y_test)
            dknn=KNN(X_train, X_test, y_train, y_test)
            dsvm=SVM(X_train, X_test, y_train, y_test)
            dn=n(X_train, X_test, y_train, y_test)
            # buscar="coronavirus"
            # lenguaje="es"
            # numero=int("10")
            # frsul=[]
            print("aquii!2")
            #render_template('index.html',dt=dt,frsul=frsul)
            # dt=API(buscar,lenguaje,numero)
            name={"ACCURACY":"0","PRESICION":"3","RECALL":"4","F-MEASURE":"7"}
        return render_template('vidrio.html',dt=dt,dnb=dnb,drf=drf,dknn=dknn,dsvm=dsvm,dn=dn)

def API(palabra,lenguaje,numero):
    a=["aqui","hola"]
    return a


def RL(X_train, X_test, y_train, y_test ):
    ###################REGRESION LOGISTICA

    #Modelo de Regresión Logística
    log = LogisticRegression(multi_class='auto',solver='lbfgs')
    log.fit(X_train, y_train)
    y_pred4 = log.predict(X_test)
    RLA="ACCURACY    :"+str(metrics.accuracy_score(y_test, y_pred4))
    RLP="PRECISION   :"+str(metrics.precision_score(y_test, y_pred4, average='macro' ,zero_division=1))
    RLR="RECALL    :"+str(metrics.recall_score(y_test, y_pred4, average='macro' ,zero_division=1))
    RLF="F-MEASURE    :"+str(metrics.f1_score(y_test, y_pred4, average='macro', zero_division=1))
    print("----------------REGRESION LOGISTICA-----------------")
    print("Precision 1 Logística: {}".format(log.score(X_train, y_train)))
    print("Accuracy Logistica:",metrics.accuracy_score(y_test, y_pred4))
    print("Recall Logistica:",metrics.recall_score(y_test, y_pred4, average='macro' ,zero_division=1))
    print("Precision 2 Logistica:",metrics.precision_score(y_test, y_pred4, average='macro' ,zero_division=1))
    print("F1-MEASURE Logistica:",metrics.f1_score(y_test, y_pred4, average='macro', zero_division=1))
    print("")
    return RLA,RLP,RLR,RLF


def NB(X_train, X_test, y_train, y_test ):
    ###################REGRESION LOGISTICA

    #Modelo de Regresión Logística
    nb=GaussianNB()
    nb.fit(X_train,y_train)
    y_pred3=nb.predict(X_test)
    RLA="ACCURACY    :"+str(metrics.accuracy_score(y_test, y_pred3))
    RLP="PRECISION    :"+str(metrics.precision_score(y_test, y_pred3, average='macro' ,zero_division=1))
    RLR="RECALL    :"+str(metrics.recall_score(y_test, y_pred3, average='macro' ,zero_division=1))
    RLF="F-MEASURE    :"+str(metrics.f1_score(y_test, y_pred3, average='macro', zero_division=1))
    print("--------------------NAIVE BAYES----------------------")
    print("Accuracy GaussianNB:",metrics.accuracy_score(y_test, y_pred3))
    print("Recall Naive Bayes:",metrics.recall_score(y_test, y_pred3, average='macro' ,zero_division=1))
    print("Precision Naive Bayes:",metrics.precision_score(y_test, y_pred3, average='macro' ,zero_division=1))
    print("F1-MEASURE Naive Bayes:",metrics.f1_score(y_test, y_pred3, average='macro', zero_division=1))
    print("")
    return RLA,RLP,RLR,RLF

def RF(X_train, X_test, y_train, y_test ):
    rf=RandomForestClassifier()
    rf.fit(X_train,y_train)
    y_pred2=rf.predict(X_test)
    RLA="ACCURACY    :"+str(metrics.accuracy_score(y_test, y_pred2))
    RLP="PRECISION   :"+str(metrics.precision_score(y_test, y_pred2, average='macro' ,zero_division=1))
    RLR="RECALL    :"+str(metrics.recall_score(y_test, y_pred2, average='macro' ,zero_division=1))
    RLF="F-MEASURE    :"+str(metrics.f1_score(y_test, y_pred2, average='macro', zero_division=1))
    print("-------------------RANDOM FOREST---------------------")
    print("Accuracy RandomForestClassifier:",metrics.accuracy_score(y_test, y_pred2))
    print("Recall RandomForest:",metrics.recall_score(y_test, y_pred2, average='macro' ,zero_division=1))
    print("Precision RandomForest:",metrics.precision_score(y_test, y_pred2, average='macro' ,zero_division=1))
    print("F1-MEASURE RandomForest:",metrics.f1_score(y_test, y_pred2, average='macro', zero_division=1))
    print("")

    return RLA,RLP,RLR,RLF

def KNN(X_train, X_test, y_train, y_test ):
    #############################KNN########################################

    knn = KNeighborsClassifier(n_neighbors =5, metric='minkowski', p=2)
    knn.fit(X_train,y_train)
    y_pred1=knn.predict(X_test)
    RLA="ACCURACY    :"+str(metrics.accuracy_score(y_test, y_pred1))
    RLP="PRECISION   :"+str(metrics.precision_score(y_test, y_pred1, average='macro' ,zero_division=1))
    RLR="RECALL    :"+str(metrics.recall_score(y_test, y_pred1, average='macro' ,zero_division=1))
    RLF="F-MEASURE    :"+str(metrics.f1_score(y_test, y_pred1, average='macro', zero_division=1))
    print("------------------------KNN--------------------------")
    print("Accuracy KNN:",metrics.accuracy_score(y_test, y_pred1))
    print("Recall KNN:",metrics.recall_score(y_test, y_pred1, average='macro' ,zero_division=1))
    print("Precision KNN:",metrics.precision_score(y_test, y_pred1, average='macro', zero_division=1))
    print("F1-MEASURE KNN:",metrics.f1_score(y_test, y_pred1, average='macro', zero_division=1))

    print("REPORTE:",metrics.classification_report(y_test, y_pred1))
    print("")

    return RLA,RLP,RLR,RLF

def SVM(X_train, X_test, y_train, y_test ):
    ########################MAQUINA DE SOPORTE VECTORIAL
    msv = SVC()
    msv.fit(X_train, y_train)
    y_pred5 = msv.predict(X_test)
    RLA="ACCURACY    :"+str(metrics.accuracy_score(y_test, y_pred5))
    RLP="PRECISION   :"+str(metrics.precision_score(y_test, y_pred5, average='macro' ,zero_division=1))
    RLR="RECALL    :"+str(metrics.recall_score(y_test, y_pred5, average='macro' ,zero_division=1))
    RLF="F-MEASURE    :"+str(metrics.f1_score(y_test, y_pred5, average='macro', zero_division=1))
    print("-------------MAQUINA DE SOPORTE VECTORIAL------------")
    print('Precisión 1 MSV: {}'.format(msv.score(X_train, y_train)))
    print("Accuracy MSV:",metrics.accuracy_score(y_test, y_pred5))
    print("Recall MSV:",metrics.recall_score(y_test, y_pred5, average='macro' ,zero_division=1))
    print("Precision 2 MSV:",metrics.precision_score(y_test, y_pred5, average='macro' ,zero_division=1))
    print("F1-MEASURE MSV:",metrics.f1_score(y_test, y_pred5, average='macro', zero_division=1))
    print("")

    return RLA,RLP,RLR,RLF
def n(X_train, X_test, y_train, y_test):

    print("-------------RED NEURONAL------------")
    mlp=MLPClassifier(hidden_layer_sizes=(10,10,10,10), max_iter=900, alpha=0.0001,
                        solver='adam', random_state=21,tol=1e-4)
    mlp.fit(X_train, y_train)
    Y_pred =mlp.predict(X_test)
    #print(classification_report(y_test,Y_pred))
    RLA="ACCURACY    :"+str(metrics.accuracy_score(y_test, Y_pred))
    RLP="PRECISION   :"+str(metrics.precision_score(y_test, Y_pred, average='macro' ,zero_division=1))
    RLR="RECALL    :"+str(metrics.recall_score(y_test, Y_pred, average='macro' ,zero_division=1))
    RLF="F-MEASURE    :"+str(metrics.f1_score(y_test, Y_pred, average='macro', zero_division=1))
    print("Accuracy RED NEURONAL:",metrics.accuracy_score(y_test, Y_pred))
    print("Recall RED NEURONAL:",metrics.recall_score(y_test, Y_pred, average='macro' ,zero_division=1))
    print("Precision RED NEURONAL:",metrics.precision_score(y_test, Y_pred, average='macro' ,zero_division=1))
    print("F1-MEASURE RED NEURONAL:",metrics.f1_score(y_test, Y_pred, average='macro', zero_division=1))
    print("")
    return RLA,RLP,RLR,RLF
if __name__=='__main__':
 app.run(port=3000,debug=True)




  