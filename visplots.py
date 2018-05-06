
# coding: utf-8

# In[1]:

import numpy as np
import plotly.graph_objs as go

from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics

def rf_organise_scores(grid_cv_scores, n_estimators, max_depth):
    # reorganisig the scores in a matrix
    scores = np.zeros((len(n_estimators), len(max_depth)))

    for score in grid_cv_scores:
        ne = score[0]['n_estimators']
        md = score[0]['max_depth']
        i = np.argmax(n_estimators == ne)
        j = np.argmax(max_depth == md)
        scores[i,j] = score[1]
    return scores


def dtDecisionPlot(XTrain, yTrain, XTest, yTest, header, feature_x=0, feature_y=1, **kwargs):
    Xtrain = XTrain[:,:2]
    h = .02  # step size in the mesh

    clf = DecisionTreeClassifier(random_state=1, **kwargs)
    clf.fit(Xtrain, yTrain)

    x_min, x_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
    y_min, y_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),  np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    trace1 = go.Contour(
        x = np.arange(x_min, x_max, h),
        y = np.arange(y_min, y_max, h),
        z = Z.reshape(xx.shape),
        showscale=False,
        opacity=0.8,
        xaxis=header[feature_x],
        yaxis=header[feature_y],
        line = dict(
            width = 1,
            color = 'black'
        ),
        colorscale=[[0, '#1976d2'], [1, '#ffcc80']],  # custom colorscale
    )

    trace2 = go.Scatter(
        x = XTest[yTest == 0, feature_x],
        y = XTest[yTest == 0, feature_y],
        mode = 'markers',
        marker = Marker(
            color = 'blue',
            line = dict(
                width = 0.9,
            )
        ),
        name = 'non-returning customers (test)'
    )

    trace3 = go.Scatter(
        x = XTest[yTest == 1, feature_x],
        y = XTest[yTest == 1, feature_y],
        mode = 'markers',
        marker = Marker(
            color = 'orange',
            line = dict(
                width = 0.9,
            ),
            symbol = 4
        ),
        name = 'returning customers (test)',
    )

    trace4 = go.Scatter(
        x = XTrain[yTrain == 0, feature_x],
        y = XTrain[yTrain == 0, feature_y],
        mode = 'markers',
        marker = Marker(
            color = 'blue',
            line = dict(
                width = 0.9,
            )
        ),
        name = 'non-returning customers (train)'
    )

    trace5 = go.Scatter(
        x = XTrain[yTrain == 1, feature_x],
        y = XTrain[yTrain == 1, feature_y],
        mode = 'markers',
        marker = Marker(
            color = 'orange',
            line = dict(
                width = 0.9,
            ),
            symbol = 4
        ),
        name = 'returning customers (train)'
    )

    layout = go.Layout(
        title = "2-Class classification Decision Trees",
        xaxis = dict(title = header[feature_x]),
        yaxis = dict(title = header[feature_y]),
        showlegend=True,
        autosize=False,
        width=700,
        height=500,
        margin=Margin(
            l=50,
            r=50,
            b=100,
            t=50,
            pad=4
        ),
    )

    data = [trace1, trace2, trace3, trace4, trace5]

    fig = dict(data=data, layout=layout)
    iplot(fig)


def rfDecisionPlot(XTrain, yTrain, XTest, yTest, header, feature_x=0, feature_y=1, **kwargs):
    Xtrain = XTrain[:, :2]
    h = .02  # step size in the mesh

    clf = RandomForestClassifier(random_state=1, **kwargs)
    clf.fit(Xtrain, yTrain)

    x_min, x_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
    y_min, y_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),  np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    trace1 = go.Contour(
        x = np.arange(x_min, x_max, h),
        y = np.arange(y_min, y_max, h),
        z = Z.reshape(xx.shape),
        showscale=False,
        opacity=0.8,
        xaxis=header[feature_x],
        yaxis=header[feature_y],
        line = dict(
            width = 1,
            color = 'black'
        ),
        colorscale=[[0, '#1976d2'], [1, '#ffcc80']],  # custom colorscale
    )

    trace2 = go.Scatter(
        x = XTest[yTest == 0, feature_x],
        y = XTest[yTest == 0, feature_y],
        mode = 'markers',
        marker = Marker(
            color = 'blue',
            line = dict(
                width = 0.9,
            )
        ),
        name = 'non-returning customers (test)'
    )

    trace3 = go.Scatter(
        x = XTest[yTest == 1, feature_x],
        y = XTest[yTest == 1, feature_y],
        mode = 'markers',
        marker = Marker(
            color = 'orange',
            line = dict(
                width = 0.9,
            ),
            symbol = 4
        ),
        name = 'returning customers (test)',
    )

    trace4 = go.Scatter(
        x = XTrain[yTrain == 0, feature_x],
        y = XTrain[yTrain == 0, feature_y],
        mode = 'markers',
        marker = Marker(
            color = 'blue',
            line = dict(
                width = 0.9,
            )
        ),
        name = 'non-returning customers (train)'
    )

    trace5 = go.Scatter(
        x = XTrain[yTrain == 1, feature_x],
        y = XTrain[yTrain == 1, feature_y],
        mode = 'markers',
        marker = Marker(
            color = 'orange',
            line = dict(
                width = 0.9,
            ),
            symbol = 4
        ),
        name = 'returning customers (train)'
    )

    layout = go.Layout(
        title = "2-Class classification Random Forests",
        xaxis = dict(title = header[feature_x]),
        yaxis = dict(title = header[feature_y]),
        showlegend=True,
        autosize=False,
        width=700,
        height=500,
        margin=Margin(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
    )

    data = [trace1, trace2, trace3, trace4, trace5]

    fig = dict(data=data, layout=layout)
    iplot(fig)

def logregDecisionPlot(XTrain, yTrain, XTest, yTest, header, feature_x=0, feature_y=1, **kwargs):
    Xtrain = XTrain[:,[feature_x, feature_y]]
    h = .02  # step size in the mesh

    clf = LogisticRegression(random_state=1, **kwargs)
    clf.fit(Xtrain, yTrain)

    x_min, x_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
    y_min, y_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),  np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    trace1 = go.Contour(
        x = np.arange(x_min, x_max, h),
        y = np.arange(y_min, y_max, h),
        z = Z.reshape(xx.shape),
        showscale=False,
        opacity=0.8,
        xaxis=header[feature_x],
        yaxis=header[feature_y],
        line = dict(
            width = 1,
            color = 'black'
        ),
        colorscale=[[0, '#1976d2'], [1, '#ffcc80']],  # custom colorscale
    )

    trace2 = go.Scatter(
        x = XTest[yTest == 0, feature_x],
        y = XTest[yTest == 0, feature_y],
        mode = 'markers',
        marker = Marker(
            color = 'blue',
            line = dict(
                width = 0.9,
            )
        ),
        name = 'non-returning customers (test)'
    )

    trace3 = go.Scatter(
        x = XTest[yTest == 1, feature_x],
        y = XTest[yTest == 1, feature_y],
        mode = 'markers',
        marker = Marker(
            color = 'orange',
            line = dict(
                width = 0.9,
            ),
            symbol = 4
        ),
        name = 'returning customers (test)',
    )

    trace4 = go.Scatter(
        x = XTrain[yTrain == 0, feature_x],
        y = XTrain[yTrain == 0, feature_y],
        mode = 'markers',
        marker = Marker(
            color = 'blue',
            line = dict(
                width = 0.9,
            )
        ),
        name = 'non-returning customers (train)'
    )

    trace5 = go.Scatter(
        x = XTrain[yTrain == 1, feature_x],
        y = XTrain[yTrain == 1, feature_y],
        mode = 'markers',
        marker = Marker(
            color = 'orange',
            line = dict(
                width = 0.9,
            ),
            symbol = 4
        ),
        name = 'returning customers (train)'
    )

    layout = go.Layout(
        title = "2-Class classification Logistic Regression",
        xaxis = dict(title = header[feature_x]),
        yaxis = dict(title = header[feature_y]),
        showlegend=True,
        autosize=False,
        width=700,
        height=500,
        margin=Margin(
            l=50,
            r=50,
            b=100,
            t=50,
            pad=4
        ),
    )

    data = [trace1, trace2, trace3, trace4, trace5]

    fig = dict(data=data, layout=layout)
    iplot(fig)

def svmDecisionPlot(XTrain, yTrain, XTest, yTest, header, feature_x=0, feature_y=1, **kwargs):
    Xtrain = XTrain[:,[feature_x, feature_y]]
    h = .02  # step size in the mesh

    clf = SVC(random_state=1, **kwargs)
    clf.fit(Xtrain, yTrain)

    x_min, x_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
    y_min, y_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),  np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    trace1 = go.Contour(
        x = np.arange(x_min, x_max, h),
        y = np.arange(y_min, y_max, h),
        z = Z.reshape(xx.shape),
        showscale=False,
        opacity=0.8,
        xaxis=header[feature_x],
        yaxis=header[feature_y],
        line = dict(
            width = 1,
            color = 'black'
        ),
        colorscale=[[0, '#1976d2'], [1, '#ffcc80']],  # custom colorscale
    )

    trace2 = go.Scatter(
        x = XTest[yTest == 0, feature_x],
        y = XTest[yTest == 0, feature_y],
        mode = 'markers',
        marker = Marker(
            color = 'blue',
            line = dict(
                width = 0.9,
            )
        ),
        name = 'non-returning customers (test)'
    )

    trace3 = go.Scatter(
        x = XTest[yTest == 1, feature_x],
        y = XTest[yTest == 1, feature_y],
        mode = 'markers',
        marker = Marker(
            color = 'orange',
            line = dict(
                width = 0.9,
            ),
            symbol = 4
        ),
        name = 'returning customers (test)',
    )

    trace4 = go.Scatter(
        x = XTrain[yTrain == 0, feature_x],
        y = XTrain[yTrain == 0, feature_y],
        mode = 'markers',
        marker = Marker(
            color = 'blue',
            line = dict(
                width = 0.9,
            )
        ),
        name = 'non-returning customers (train)'
    )

    trace5 = go.Scatter(
        x = XTrain[yTrain == 1, feature_x],
        y = XTrain[yTrain == 1, feature_y],
        mode = 'markers',
        marker = Marker(
            color = 'orange',
            line = dict(
                width = 0.9,
            ),
            symbol = 4
        ),
        name = 'returning customers (train)'
    )

    layout = go.Layout(
        title = "2-Class classification SVMs",
        xaxis = dict(title = header[feature_x]),
        yaxis = dict(title = header[feature_y]),
        showlegend=True,
        autosize=False,
        width=700,
        height=500,
        margin=Margin(
            l=50,
            r=50,
            b=100,
            t=50,
            pad=4
        ),
    )

    data = [trace1, trace2, trace3, trace4, trace5]

    fig = dict(data=data, layout=layout)
    iplot(fig)

def rfAvgAcc(rfModel, XTest, yTest):
    preds = []
    avgPred = []
    df = []

    for i,tree in enumerate(rfModel.estimators_):
        predTree = tree.predict(XTest)
        accTree  = round(metrics.accuracy_score(yTest, predTree),2)
        preds.append(accTree)
        if i==0:
            df = predTree
        else:
            df = np.vstack((df,predTree))

    for j in np.arange(df.shape[0]):
        j=j+1
        mv = []
        for i in np.arange(df.shape[1]):
            (values,counts) = np.unique(df[:j,i],return_counts=True)
            ind=np.argmax(counts)
            mv.append(values[ind].astype(int))
        avgPred.append(metrics.accuracy_score(yTest, mv))

    trace = go.Scatter(
        y=avgPred,
        x=np.arange(df.shape[0]),
        mode='markers+lines',
        name = "Ensemble accuracy trend"
    )

    layout = go.Layout(
        title = "Ensemble accuracy over increasing number of trees",
        xaxis = dict(title = "Number of trees", nticks = 15),
        yaxis = dict(title = "Accuracy"),
        showlegend=False,
        autosize=False,
        width=1000,
        height=500,
        margin=Margin(
            l=70,
            r=50,
            b=100,
            t=50,
            pad=4
        ),
    )

    data = [trace]

    fig = dict(data=data, layout=layout)
    iplot(fig)


# In[ ]:



