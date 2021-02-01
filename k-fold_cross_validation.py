import matplotlib. pyplot as plt
import numpy as np


data_train = {'X': np.genfromtxt('data_train_X.csv', delimiter= ','),
                                 't': np.genfromtxt('data_train_y.csv', delimiter=',')}
data_test = {'X': np.genfromtxt('data_test_X.csv', delimiter= ','),
                                 't': np.genfromtxt('data_test_y.csv', delimiter=',')}

t_train = data_train['t']
X_train = data_train['X']

t_test = data_test['t']
X_test = data_test['X']

data_train_ = []
for i in range(len(t_train)):
    data_train_.append((t_train[i], X_train[i]))

data_test_ = []
for i in range(len(t_test)):
    data_test_.append((t_test[i], X_test[i]))

def shuffle_data(data):
    np.random.shuffle(data)
    return data

def split_data(data, num_folds, fold): #data is [(2, array([2,3,4])), array((b, [1,2,3]))] #fold = 1,2,3..
    data_rest = []
    data_fold = []
    slicing = len(data) / num_folds
    for i in range(len(data)):
        if i != (fold): 
            data_rest.extend(data[int(i*slicing):int((i+1)*slicing)])
        else:
            data_fold.extend(data[int(i*slicing):int((i+1)*slicing)])
    return data_fold, data_rest


def train_model(data, lambd):
    X = []
    t = []
    
    for i in range(len(data)):
        X.append(data[i][1])
        t.append(data[i][0])
    X = np.array(X)
    t = np.array(t)
    X_transpose = X.transpose()
    return (np.linalg.inv(X_transpose.dot(X) + lambd * np.identity(X.shape[1])).dot(X_transpose)).dot(t)
    
def predict(data, model):
    X = []
    t = []
    for i in range(len(data)):
        X.append(data[i][1])
        t.append(data[i][0])
    X_np = np.array(X)
    return np.dot(X_np, model)
    
def loss(data, model):
    X = []
    t = []
    n = len(data)
    for i in range(len(data)):
        X.append(data[i][1])
        t.append(data[i][0])
    X = np.array(X)
    t = np.array(t)
    prediction = predict(data, model)
    error = (np.linalg.norm(t - prediction ,2) **2)/n
    return error

def cross_validation(data, num_folds, lambd_seq):
    data = shuffle_data(data)
    cv_error = []
    for i in range(len(lambd_seq)):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0
        for fold in range(num_folds):
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, lambd)      
            cv_loss_lmd += loss(val_cv, model)
        cv_error.append(cv_loss_lmd / num_folds)
    return cv_error

if __name__ == '__main__':
    
    lambd_seq = np.linspace (0.02 , 1.5 , num =50)
    training_data = shuffle_data(data_train_)
    testing_data = shuffle_data(data_test_)  
    training_error_list = []
    testing_error_list = []
    
    five_fold_error = cross_validation(training_data, 5, lambd_seq)
    ten_fold_error = cross_validation(training_data, 10, lambd_seq)
    
    for lambd in lambd_seq:
        model = train_model(training_data, lambd)
        
        training_error = loss(training_data, model)
        training_error_list.append(training_error)
        
        testing_error = loss(testing_data, model)
        testing_error_list.append(testing_error)
        
        print("training error is "+ str(training_error) + " for lamda = " +str (lambd))
        print("testing error is "+ str(testing_error) + " for lamda = " +str (lambd)+"\n")
        
        
    plt.plot(lambd_seq, training_error_list, label='training_error') 
    plt.plot(lambd_seq, testing_error_list, label='testing_error')
    plt.plot(lambd_seq, five_fold_error, label='five_fold_error') 
    plt.plot(lambd_seq, ten_fold_error, label='ten_fold_error')
    
    plt.ylabel("Error")
    plt.xlabel("lambda")
    plt.legend()
    
    min_five_fold = min(five_fold_error)
    min_ten_fold = min(ten_fold_error)
    print("Lambda value proposed by 5-fold cv is: " + str(lambd_seq[five_fold_error.index(min_five_fold)]) + 
          " having a cv error of " + str(min_five_fold))
    print("Lambda value proposed by 10-fold cv is: " + str(lambd_seq[ten_fold_error.index(min_ten_fold)]) + 
          " having a cv error of " + str(min_ten_fold))
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    