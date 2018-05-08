# Value network for 2 player board games. Implemented as a simple feed-forward CNN.
# See mcts.py for sample usage.
#
# Author: Prithvijit Chakrabarty (prithvichakra@gmail.com)

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

def start_sess():
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    return sess

#Model to measure state value
class Q():
    def __init__(self,arch):
        
        #Default hyperparams
        self.lr = 1e-4
        self.eps = 1e-8
        self.bz = 100
        self.epch = 1000
        self.tc = 1
        
        inshp = arch['in_d']
        in_dim = np.prod(arch['in_d'])
        self.x = tf.placeholder(tf.float32,shape=[None,in_dim])
        self.y = tf.placeholder(tf.float32,shape=[None,1])
        self.keep_prob = tf.placeholder(tf.float32)

        nfilt = arch['nfilt']
        out = tf.reshape(self.x,[-1,inshp[0],inshp[1],inshp[2]])
        out = tf.layers.conv2d(inputs=out,filters=nfilt,kernel_size=[3,3],padding='same',activation=tf.nn.relu,strides=1)
        f_dim = np.prod(out.get_shape().as_list()[1:])
        out = tf.reshape(out,[-1,f_dim])

        w1 = tf.get_variable('w1',shape=[f_dim,arch['h1']],initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable('b1',shape=[arch['h1']],initializer=tf.zeros_initializer())
        out = tf.nn.relu(tf.matmul(out,w1)+b1)
        out = tf.nn.dropout(out,self.keep_prob)
        w2 = tf.get_variable('w2',shape=[arch['h1'],arch['h2']],initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2',shape=[arch['h2']],initializer=tf.zeros_initializer())
        out = tf.nn.relu(tf.matmul(out,w2)+b2)
        out = tf.nn.dropout(out,self.keep_prob)
        w3 = tf.get_variable('w3',shape=[arch['h2'],1],initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable('b3',shape=[1],initializer=tf.zeros_initializer())
        out = tf.nn.sigmoid(tf.matmul(out,w3)+b3)

        self.out = out
        self.loss = tf.losses.mean_squared_error(self.y,self.out)
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.lr,epsilon=self.eps).minimize(self.loss)

    #Method to run training
    def train(self,dsfile,model_path):
        ds = list(np.load(dsfile))
        ds = [(t.flatten(),g) for t,g in ds]
        np.random.shuffle(ds)
        n = len(ds)
        tds = ds[:int(self.tc*n)]
        x,y = map(np.array,zip(*tds))
        y = y.reshape((y.shape[0],1))
        vds = ds[int(self.tc*n):]
        vx,vy = map(np.array,zip(*vds))
        vy = vy.reshape((vy.shape[0],1))
        print 'train ds shape:',x.shape,'--',y.shape
        print 'val ds shape:',vx.shape,'--',vy.shape

        sess = start_sess()
        saver = tf.train.Saver()
        loss_hist = []
        val_hist = []
        for epoch in range(self.epch):
            #Training step and loss
            epch_loss = []
            for bi in range(0,len(tds)-self.bz,self.bz):
                bx,by = x[bi:bi+self.bz],y[bi:bi+self.bz]
                feed_dict = {self.x:bx,self.y:by,self.keep_prob:0.5}
                _,loss,py = sess.run([self.train_step,self.loss,self.out],feed_dict=feed_dict)
                epch_loss.append(loss)
            loss_hist.append(np.mean(epch_loss))
            #Validation loss
            epch_val_loss = []
            for bi in range(0,len(vds)-self.bz,self.bz):
                bx,by = vx[bi:bi+self.bz],vy[bi:bi+self.bz]
                vloss = sess.run(self.loss,feed_dict={self.x:bx,self.y:by,self.keep_prob:0.5})
                epch_val_loss.append(vloss)
            val_hist.append(np.mean(epch_val_loss))
            print 'Epoch:',str(epoch)+': \t\tloss: '+str(loss_hist[-1])+'\t\tval loss: '+str(val_hist[-1])
        plt.plot(loss_hist,label='Training loss')
        plt.plot(val_hist,label='Validation loss')
        plt.legend()
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
        py = self.sess.run(self.out,feed_dict={self.x:vx,self.keep_prob:1})
        return py[0]

    def close(self):
        self.sess.close()
