class SparseKernalLogisticRegression:
    def __init__(self,X,y,kernel="drgbf"):
        self.X = X
        self.y = y
        if kernel == "drgbf":
            self.kernel = self.dGRBF
        else:
            self.kernel = kernel
        self.K = self.getSparseCov(X)
    def dGRBF(self,d):
        return np.exp(-(d*d)/2)
    def getSparseCov(self,X):
        self.neigh =  NearestNeighbors(n_neighbors=90,n_jobs=-1)
        self.neigh.fit(X)
        neighbours=self.neigh.kneighbors(X)
        (values,indices) = neighbours
        neighbours = []
        for i in range(len(indices)):
            neighbours.append({})
        for i in range(len(indices)):
            for j in range(len(indices[i])):
                neighbours[indices[i][j]][i]=neighbours[i][indices[i][j]]=self.kernel(values[i][j])
        values=[]
        indices=[]
        for i in range(len(X)):
            for j in neighbours[i]:
                indices.append([i,j])
                values.append(neighbours[i][j])
        with tf.device('/GPU:0'):
            return tf.dtypes.cast(tf.SparseTensor(indices=indices,values=values,dense_shape=[len(X),len(X)]), tf.float32)
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def y_pred(self,K,alpha):
        with tf.device('/GPU:0'):
            return np.ndarray.flatten(self.sigmoid(tf.sparse.sparse_dense_matmul(K, tf.convert_to_tensor(alpha.reshape(len(alpha),1) ,tf.float32)).numpy()))
    def loglikelyhood(self,alpha):
        __y = self.y_pred(self.K,alpha)
        print((__y==0).sum())
        print((__y==1).sum())
        return self.y.dot(np.log(__y))+(1-self.y).dot(np.log(1-__y))+(self.lamda/2*(alpha.T@np.ndarray.flatten(self.sigmoid(tf.sparse.sparse_dense_matmul(self.K, tf.convert_to_tensor(alpha.reshape(len(alpha),1) ,tf.float32)).numpy()))))
    def grad(self,alpha):
        __y = self.y_pred(self.K,alpha)
        with tf.device('/GPU:0'):
            return tf.sparse.sparse_dense_matmul(self.K, tf.convert_to_tensor((self.y-__y-(self.lamda*alpha)).reshape(len(alpha),1) ,tf.float32)).numpy().reshape(self.K.shape[0])
    def fit(self,lamda):
        self.lamda=lamda
        alpha = np.zeros(len(self.X))
        alpha = optimize.minimize(
                                fun=lambda a: -self.loglikelyhood(a),
                                x0=alpha.copy(), 
                                method='CG', 
                                jac=lambda a: -self.grad(a), 
                                ####options={'eps':1e-15}
                                ).x
        self.alpha = alpha
    def predict_proba(self,X_test):
        neighbours=self.neigh.kneighbors(X_test)
        (values,indices) = neighbours
        neighbours = []
        for i in range(len(indices)):
            neighbours.append({})
        for i in range(len(indices)):
            for j in range(len(indices[i])):
                neighbours[i][indices[i][j]]=self.kernel(values[i][j])
        values=[]
        indices=[]
        for i in range(len(X_test)):
            for j in neighbours[i]:
                indices.append([i,j])
                values.append(neighbours[i][j])
        with tf.device('/GPU:0'):
            return self.y_pred(
                tf.dtypes.cast(tf.SparseTensor(indices=indices,values=values,dense_shape=[len(X_test),len(self.X)]), tf.float32),
                self.alpha
            )
    def predict(self,X_test,b=0):
        y_test = self.predict_proba(X_test)
        return np.where(y_test<b,0,1)