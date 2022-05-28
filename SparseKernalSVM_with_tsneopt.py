class SparseKernelSvmClassifier:    
    def __init__(self, C, kernel):
        self.C = C                               
        self.kernel = kernel          # <---
        self.alpha = None
        self.supportVectors = None
    def getSparseCov(self,X,y):
        self.neigh =  NearestNeighbors(n_neighbors=90,n_jobs=-1)
        self.neigh.fit(X)
        neighbours=self.neigh.kneighbors(X)
        (values,indices) = neighbours
        neighbours = []
        for i in range(len(indices)):
            neighbours.append({})
        for i in range(len(indices)):
            for j in range(len(indices[i])):
                neighbours[indices[i][j]][i]=neighbours[i][indices[i][j]]=y[i]*y[indices[i][j]]*self.kernel(X[i],X[j])
        values=[]
        indices=[]
        for i in range(len(X)):
            for j in neighbours[i]:
                indices.append([i,j])
                values.append(neighbours[i][j])
        with tf.device('/GPU:0'):
            return tf.dtypes.cast(tf.SparseTensor(indices=indices,values=values,dense_shape=[len(X),len(X)]), tf.float32)   
    def fit(self, X, y):
        N = len(y)
        self.N=N
        self.X = X
        ##y = tranform(Y)
        G = self.getSparseCov(X,y)
        # Lagrange dual problem
        def Ld0(G, alpha):
            return alpha.sum() - 0.5 * (alpha.T @ (tf.sparse.sparse_dense_matmul(G, tf.convert_to_tensor(alpha.reshape(len(alpha),1) ,tf.float32)).numpy().reshape(len(alpha))))
 
        # Partial derivate of Ld on alpha
        def Ld0dAlpha(G, alpha):
            with tf.device('/GPU:0'):
                return np.ones_like(alpha) - (tf.sparse.sparse_dense_matmul(G, tf.convert_to_tensor(alpha.reshape(len(alpha),1) ,tf.float32)).numpy().reshape(len(alpha)))
        constraints = ({'type': 'eq',   'fun': lambda a: np.dot(a, y),     'jac': lambda a: y})

        # Maximize by minimizing the opposite
        optRes = optimize.minimize(fun=lambda a: -Ld0(G, a),
                                   x0=np.ones(N), 
                                   method='SLSQP', 
                                   bounds=np.array((0,self.C)*N).reshape(N,2),
                                   jac=lambda a: -Ld0dAlpha(G, a), 
                                   constraints=constraints)
        self.alpha = optRes.x
        # --->
        epsilon = 1e-8
        self.a = np.where(self.alpha<epsilon,0,self.alpha).reshape(N)
        self.ay = np.multiply(y.reshape(N),self.a)
        supportIndices = self.alpha > epsilon
        self.supportVectors = X[supportIndices]
        self.supportAlphaY = y[supportIndices] * self.alpha[supportIndices]
        # <---
    def get_d1(self,X):
        neighbours=self.neigh.kneighbors(X)
        (values,indices) = neighbours
        neighbours = []
        for i in range(len(indices)):
            neighbours.append({})
        for i in range(len(indices)):
            for j in range(len(indices[i])):
                neighbours[i][indices[i][j]]=self.kernel(X[i],self.X[indices[i][j]])
        values=[]
        indices=[]
        for i in range(len(X)):
            for j in neighbours[i]:
                indices.append([i,j])
                values.append(neighbours[i][j])
        with tf.device('/GPU:0'):
            return tf.sparse.sparse_dense_matmul(
                tf.dtypes.cast(tf.SparseTensor(indices=indices,values=values,dense_shape=[len(X),self.N]), tf.float32),
                tf.dtypes.cast(self.ay.reshape(self.N,1), tf.float32)
            ).numpy().reshape(len(X))

    def get_d2(self,X):
        def predict1(x):
            x1 = np.apply_along_axis(lambda s: self.kernel(s,x), 1, self.supportVectors)
            x2 = x1 * self.supportAlphaY
            return np.sum(x2)        
        return np.apply_along_axis(predict1, 1, X)

    def predict(self, X):
        if(len(self.supportAlphaY)>90):
            return 2*(self.get_d1(X)>0)-1
        return 2*(self.get_d2(X)>0)-1
    def predict_proba(self,X):
        if(len(self.supportAlphaY)>90):
            return 1/(1+np.exp(-self.get_d1(X)))
        return 1/(1+np.exp(-self.get_d2(X)))