import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

def start_sess():
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    return sess

class Q():
    def __init__(self,arch):
        
        self.lr = 1e-3
        self.eps = 1e-8
        self.bz = 100
        self.epch = 400
        self.tc = 1
        self.vc = 0
        
        self.x = tf.placeholder(tf.float32,shape=[None,arch['in_d']])
        self.y = tf.placeholder(tf.float32,shape=[None,1])
        out = self.x
        w1 = tf.get_variable('w1',shape=[arch['in_d'],arch['h1']],initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable('b1',shape=[arch['h1']],initializer=tf.zeros_initializer())
        out = tf.nn.relu(tf.matmul(out,w1)+b1)
        out = tf.nn.dropout(out,0.5)
        w2 = tf.get_variable('w2',shape=[arch['h1'],arch['h2']],initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2',shape=[arch['h2']],initializer=tf.zeros_initializer())
        out = tf.nn.relu(tf.matmul(out,w2)+b2)
        out = tf.nn.dropout(out,0.5)
        w3 = tf.get_variable('w3',shape=[arch['h2'],1],initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable('b3',shape=[1],initializer=tf.zeros_initializer())
        out = tf.nn.sigmoid(tf.matmul(out,w3)+b3)

        self.out = out
        self.loss = tf.losses.mean_squared_error(self.y,self.out)
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.lr,epsilon=self.eps).minimize(self.loss)

    def train(self,dsfile,model_path):
        ds = list(np.load(dsfile))
        ds = [(t.flatten(),g) for t,g in ds]
        np.random.shuffle(ds)
        n = len(ds)
        tds = ds[:int(self.tc*n)]
        x,y = map(np.array,zip(*tds))
        y = y.reshape((y.shape[0],1))
        print x.shape,'--',y.shape

        sess = start_sess()
        saver = tf.train.Saver()
        loss_hist = []
        for epoch in range(self.epch):
            epch_loss = []
            for bi in range(0,len(tds)-self.bz,self.bz):
                bx,by = x[bi:bi+self.bz],y[bi:bi+self.bz]
                feed_dict = {self.x:bx,self.y:by}
                _,loss,py = sess.run([self.train_step,self.loss,self.out],feed_dict=feed_dict)
                #pvy = self.sess.run(self.out,feed_dict={self.x:v_x,self.keep_prob:1})
                epch_loss.append(loss)
            loss_hist.append(np.mean(epch_loss))
            print 'Epoch',str(epoch)+': \tloss '+str(loss_hist[-1])
        plt.plot(loss_hist)
        plt.show()
        saver.save(sess,model_path)
        print 'Model saved'
        sess.close()

    def load(self,model_path):
        sess = start_sess()
        saver = tf.train.Saver()
        saver.restore(sess,model_path)
        self.sess = sess

    def predict(self,state):
        vx = np.array([state.flatten()])
        py = self.sess.run(self.out,feed_dict={self.x:vx})
        return py[0]

    def close(self):
        self.sess.close()

"""model_path = './model'
dsfile = './conn4data.npy'
arch = {'in_d' : 3*5*5,
        'h1'   : 30,
        'h2'   : 30,
       }

q = Q(arch)
q.train(dsfile,model_path)"""
