import ROOT as R
import tensorflow as tf
import numpy as np
from tensorflow import keras
import json
from ROOT import TLorentzVector
import math
import os


### All the parameters:
n_tau    = 100 # number of taus (or events) per batch
n_pf     = 50 #100 # number of pf candidates per event
n_fe     = 33   # total muber of features: 24
n_labels = 4#6    # number of labels per event
n_epoch  = 100 #100  # number of epochs on which to train
n_steps_val   = 1000 #14138
n_steps_test  = 10 #63620  # number of steps in the evaluation: (events in conf_dm_mat) = n_steps * n_tau
entry_start   = 0
n_batch_training = 2000
entry_stop    = n_batch_training*100 #6362169 # total number of events in the dataset = 14'215'297
# 45% = 6'362'169
# 10% = 1'413'815 (approximative calculations have been rounded)
entry_start_val  = entry_stop +1
entry_stop_val   = entry_stop + n_tau*n_steps_val + 1
entry_start_test = entry_stop_val+1
entry_stop_test  = entry_stop_val + n_tau*n_steps_test + 1

print('\n********************************************************')
print('****************** Parameters **************************')
print('********************************************************')
print('Batch size: ', n_tau)
print('Number of batches in training: ', n_batch_training)
print('Number of batches in validation: ', n_steps_val)
print('********************************************************')
print('********************************************************')
print('********************************************************')
print('Security checks:')
print('Training start entry: ', entry_start)
print('Training stop entry: ', entry_stop)
print('Validation start entry:', entry_start_val)
print('Validation stop entry: ',entry_stop_val)
print('Test start entry: ', entry_start_test)
print('Test stop entry (<= 14138155): ',entry_stop_test,'\n')

# print('Test stop entry (<= 14215297): ',entry_stop_test,'\n')
# 45% = 6'396'973
# 10% = 1'421'351 (approximative calculations have been rounded)



class StdLayer(tf.keras.layers.Layer):
    def __init__(self, file_name, var_pos, n_sigmas, **kwargs):
        with open(file_name) as json_file:
            data_json = json.load(json_file)
        n_vars = len(var_pos)
        self.vars_std = [1] * n_vars
        self.vars_mean = [1] * n_vars
        self.vars_apply = [False] * n_vars
        for var, ms in data_json.items():
            pos = var_pos[var]
            self.vars_mean[pos]  = ms[0]['mean']
            self.vars_std[pos]   = ms[0]['std']
            self.vars_apply[pos] = True
        self.vars_mean  = tf.constant(self.vars_mean,  dtype=tf.float32)
        self.vars_std   = tf.constant(self.vars_std,   dtype=tf.float32)
        self.vars_apply = tf.constant(self.vars_apply, dtype=tf.bool)
        self.n_sigmas   = n_sigmas

        super(StdLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(StdLayer, self).build(input_shape)

    def call(self, X):
        Y = tf.clip_by_value(( X - self.vars_mean ) / self.vars_std, -self.n_sigmas, self.n_sigmas)
        return tf.where(self.vars_apply, Y, X)

class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, file_name, var_pos, interval_to_scale, **kwargs):
        with open(file_name) as json_file:
            data_json = json.load(json_file)
        self.a = interval_to_scale[0]
        self.b = interval_to_scale[1]
        n_vars = len(var_pos)
        self.vars_max   = [1] * n_vars
        self.vars_min   = [1] * n_vars
        self.vars_apply = [False] * n_vars
        for var, mm in data_json.items():
            pos = var_pos[var]
            self.vars_min[pos]   = mm[0]['min']
            self.vars_max[pos]   = mm[0]['max']
            self.vars_apply[pos] = True
        self.vars_min   = tf.constant(self.vars_min,   dtype=tf.float32)
        self.vars_max   = tf.constant(self.vars_max,   dtype=tf.float32)
        self.vars_apply = tf.constant(self.vars_apply, dtype=tf.bool)
        self.y = (self.b - self.a) / (self.vars_max - self.vars_min)

        super(ScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ScaleLayer, self).build(input_shape)

    def call(self, X):
        Y = tf.clip_by_value( (self.y * ( X - self.vars_min))  + self.a , self.a, self.b)
        return tf.where(self.vars_apply, Y, X)

class MyModel(tf.keras.Model):
    def __init__(self, map_features, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.px_index    = map_features["pfCand_px"]
        self.py_index    = map_features["pfCand_py"]
        self.pz_index    = map_features["pfCand_pz"]
        self.E_index     = map_features["pfCand_E"]
        self.valid_index = map_features["pfCand_valid"]
        self.map_features = map_features

        self.embedding1   = tf.keras.layers.Embedding(350,3)
        self.embedding2   = tf.keras.layers.Embedding(4  ,2)
        self.embedding3   = tf.keras.layers.Embedding(8  ,2)
        self.normalize    = StdLayer('mean_std.txt', map_features, 5, name='std_layer')
        self.scale        = ScaleLayer('min_max.txt', map_features, [-1,1], name='scale_layer')
        self.flatten      = tf.keras.layers.Flatten() # flattens a ND array to a 2D array

        self.n_layers = 5 #10
        self.dense = []
        self.dropout_dense = []
        self.batch_norm_dense = []
        self.acti_dense = []

        for i in range(0,self.n_layers):
            self.dense.append(tf.keras.layers.Dense(3600, name='dense_{}'.format(i)))
            self.batch_norm_dense.append(tf.keras.layers.BatchNormalization(name='batch_normalization_{}'.format(i)))
            self.acti_dense.append(tf.keras.layers.Activation('relu', name='acti_dense_{}'.format(i)))
            self.dropout_dense.append(tf.keras.layers.Dropout(0.25,name='dropout_dense_{}'.format(i)))

        self.output_layer_2   = tf.keras.layers.Dense(2  , name='output_layer_2')
        self.output_layer_100 = tf.keras.layers.Dense(100, name='output_layer_100', activation='softmax')

    # @tf.function
    def call(self, xx):
        x_em1 = self.embedding1(tf.abs(xx[:,:,self.map_features['pfCand_pdgId']]))
        x_em2 = self.embedding2(tf.abs(xx[:,:,self.map_features['pfCand_fromPV']]))
        x_em3 = self.embedding3(tf.abs(xx[:,:,self.map_features['pfCand_pvAssociationQuality']]))
        x = self.normalize(xx)
        x = self.scale(x)

        x_part1 = x[:,:,:self.map_features['pfCand_pdgId']]
        x_part2 = x[:,:,(self.map_features["pfCand_fromPV"]+1):]
        x = tf.concat((x_part1,x_part2,x_em1,x_em2,x_em3),axis = 2)

        x = self.flatten(x)
        for i in range(0,self.n_layers):
            x = self.dense[i](x)
            x = self.batch_norm_dense[i](x)
            x = self.acti_dense[i](x)
            x = self.dropout_dense[i](x)
        x2   = self.output_layer_2(x)
        x100 = self.output_layer_100(x)
        print('x100.shape: ', x100.shape)
        print('x2.shape: ', x2.shape)

        ### 4-momentum:
        mypxs  = xx[:,:,self.px_index] * x100 * xx[:,:,self.valid_index]
        mypys  = xx[:,:,self.py_index] * x100 * xx[:,:,self.valid_index]
        mypzs  = xx[:,:,self.pz_index] * x100 * xx[:,:,self.valid_index]
        myEs   = xx[:,:,self.E_index]  * x100 * xx[:,:,self.valid_index]

        mypx   = tf.reduce_sum(mypxs, axis = 1)
        mypy   = tf.reduce_sum(mypys, axis = 1)
        mypz   = tf.reduce_sum(mypzs, axis = 1)
        myE    = tf.reduce_sum(myEs , axis = 1)

        mypx2  = tf.square(mypx)
        mypy2  = tf.square(mypy)
        mypz2  = tf.square(mypz)

        mypt   = tf.sqrt(mypx2 + mypy2)
        mymass = tf.square(myE) - mypx2 - mypy2 - mypz2
        absp   = tf.sqrt(mypx2 + mypy2 + mypz2)

        ### for myeta and myphi:
        myphi = mypt*0.0
        myeta = mypt*0.0

        cosTheta = tf.where(absp==0, 1.0, mypz/absp)
        myeta = tf.where(cosTheta*cosTheta < 1, -0.5*tf.math.log((1.0-cosTheta)/(1.0+cosTheta)), 0.0)
        myphi = tf.where(tf.math.logical_and(mypx == 0, mypy == 0), 0.0, tf.math.atan2(mypy, mypx))

        xx20 = x2[:,0] # is needed doesn't like direct input
        xx21 = x2[:,1]

        xout = tf.stack([xx20,xx21,mypt,myeta,myphi,mymass], axis=1)

        return xout

class MyGNNLayer(tf.keras.layers.Layer):
    def __init__(self, n_dim, num_outputs, **kwargs):
        super(MyGNNLayer, self).__init__(**kwargs)
        self.n_dim        = n_dim
        self.num_outputs  = num_outputs
        self.supports_masking = True # to pass the mask to the next layers and not destroy it

    def build(self, input_shape):
        self.A = self.add_weight("A", shape=((input_shape[-1]+1) * 2 - 1, self.num_outputs),
                                initializer="he_uniform", trainable=True)
        self.b = self.add_weight("b", shape=(self.num_outputs,), initializer="he_uniform", trainable=True)

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], self.num_outputs]

    @tf.function
    def call(self, x, mask):
        # print('check x shape 1: ', x)
        ### a and b contain copies for each pf_Cand:
        x_shape = tf.shape(x)

        ## a tensor: a[n_tau, pf_others, pf, features]
        rep = tf.stack([1,x_shape[1],1])
        a   = tf.tile(x, rep)
        a   = tf.reshape(a,(x_shape[0],x_shape[1],x_shape[1],x_shape[2]))

        ## b tensor: a[n_tau, pf, pf_others, features]
        rep = tf.stack([1,1,x_shape[1]])
        b   = tf.tile(x, rep)
        b   = tf.reshape(b,(x_shape[0],x_shape[1],x_shape[1],x_shape[2]))


        ### Compute distances:
        ca = a[:,:,:, -self.n_dim:]
        cb = b[:,:,:, -self.n_dim:]
        c_shape = tf.shape(ca)
        diff = ca-cb
        diff = tf.math.square(diff)
        dist = tf.math.reduce_sum(diff, axis = -1)
        dist = tf.reshape(dist,(c_shape[0],c_shape[1],c_shape[2],1)) # needed to concat
        na   = tf.concat((a,dist),axis=-1) #a[n_tau, pf_others, pf, features+1]


        ### Weighted sum of features:
        w = tf.math.exp(-10*na[:,:,:,-1]) # weights
        w_shape = tf.shape(w)
        w    = tf.reshape(w,(w_shape[0],w_shape[1],w_shape[2],1)) # needed for multiplication
        mask = tf.reshape(mask, (w_shape[0],w_shape[1],1)) # needed for multiplication
        ## copies of mask:
        rep  = tf.stack([1,w_shape[1],1])
        mask_copy = tf.tile(mask, rep)
        mask_copy = tf.reshape(mask_copy,(w_shape[0],w_shape[1],w_shape[2],1))
        # mask_copy = [n_tau, n_pf_others, n_pf, mask]
        s = na * w * mask_copy # weighted na
        ss = tf.math.reduce_sum(s, axis = 1) # weighted sum of features
        # ss = [n_tau, n_pf, features+1]
        self_dist = tf.zeros((x_shape[0], x_shape[1], 1))
        xx = tf.concat([x, self_dist], axis = 2) # [n_tau, n_pf, features+1]
        ss = ss - xx # difference between weighted features and original ones
        x = tf.concat((x, ss), axis = 2) # add to original features
        # print('check x shape 2: ', x) #(n_tau, n_pf, features*2+1)


        ### Ax+b:
        output = tf.matmul(x, self.A) + self.b

        # print('output.shape: ', output.shape)
        output = output * mask # reapply mask to be sure

        return output


class MyGNN(tf.keras.Model):

    def __init__(self, mode, map_features, filename, **kwargs):
        super(MyGNN, self).__init__()
        self.map_features = map_features
        self.mode = mode
        self.filename = filename

        self.n_gnn_layers      = kwargs["n_gnn_layers"]
        self.n_dim_gnn         = kwargs["n_dim_gnn"]
        self.n_output_gnn      = kwargs["n_output_gnn"]
        self.n_output_gnn_last = kwargs["n_output_gnn_last"]
        self.n_dense_layers    = kwargs["n_dense_layers"]
        self.n_dense_nodes     = kwargs["n_dense_nodes"]
        self.wiring_mode       = kwargs["wiring_mode"]
        self.dropout_rate      = kwargs["dropout_rate"]

        self.embedding1   = tf.keras.layers.Embedding(350,2)
        self.embedding2   = tf.keras.layers.Embedding(4  ,2)
        self.embedding3   = tf.keras.layers.Embedding(8  ,2)
        self.normalize    = StdLayer('mean_std.txt', self.map_features, 5, name='std_layer')
        self.scale        = ScaleLayer('min_max.txt', self.map_features, [-1,1], name='scale_layer')

        self.GNN_layers  = []
        self.batch_norm  = []
        self.acti_gnn    = []
        self.dense            = []
        self.dense_batch_norm = []
        self.dense_acti       = []
        if(self.dropout_rate > 0):
            self.dropout_gnn = []
            self.dropout_dense    = []

        list_outputs = [self.n_output_gnn] * (self.n_gnn_layers-1) + [self.n_output_gnn_last]
        list_n_dim   = [2] + [self.n_dim_gnn] * (self.n_gnn_layers-1)
        self.n_gnn_layers = len(list_outputs)
        self.n_dense_layers = self.n_dense_layers

        for i in range(self.n_gnn_layers):
            self.GNN_layers.append(MyGNNLayer(n_dim=list_n_dim[i], num_outputs=list_outputs[i], name='GNN_layer_{}'.format(i)))
            self.batch_norm.append(tf.keras.layers.BatchNormalization(name='batch_normalization_{}'.format(i)))
            self.acti_gnn.append(tf.keras.layers.Activation("tanh", name='acti_gnn_{}'.format(i)))
            if(self.dropout_rate > 0):
                self.dropout_gnn.append(tf.keras.layers.Dropout(self.dropout_rate ,name='dropout_gnn_{}'.format(i)))

        for i in range(self.n_dense_layers-1):
            self.dense.append(tf.keras.layers.Dense(self.n_dense_nodes, kernel_initializer="he_uniform",
                                bias_initializer="he_uniform", name='dense_{}'.format(i)))
            self.dense_batch_norm.append(tf.keras.layers.BatchNormalization(name='dense_batch_normalization_{}'.format(i)))
            self.dense_acti.append(tf.keras.layers.Activation("sigmoid", name='dense_acti{}'.format(i)))
            if(self.dropout_rate > 0):
                self.dropout_dense.append(tf.keras.layers.Dropout(self.dropout_rate ,name='dropout_dense_{}'.format(i)))

        n_last = 4 if mode == "p4_dm" else 2
        self.dense2 = tf.keras.layers.Dense(n_last, kernel_initializer="he_uniform",
                                bias_initializer="he_uniform", name='dense2')




    @tf.function
    def call(self, xx):

        x_mask = xx[:,:,self.map_features['pfCand_valid']]

        x_em1 = self.embedding1(tf.abs(xx[:,:,self.map_features['pfCand_pdgId']]))
        x_em2 = self.embedding2(tf.abs(xx[:,:,self.map_features['pfCand_fromPV']]))
        x_em3 = self.embedding3(tf.abs(xx[:,:,self.map_features['pfCand_pvAssociationQuality']]))
        x = self.normalize(xx)
        x = self.scale(x)

        x_part1 = x[:,:,:self.map_features['pfCand_pdgId']]
        x_part2 = x[:,:,(self.map_features["pfCand_fromPV"]+1):]
        x = tf.concat((x_em1,x_em2,x_em3,x_part1,x_part2),axis = 2)

        if(self.wiring_mode=="m2"):
            for i in range(self.n_gnn_layers):
                if i > 1:
                    x = tf.concat([x0, x], axis=2)
                x = self.GNN_layers[i](x, mask=x_mask)
                if i == 0:
                    x0 = x
                x = self.batch_norm[i](x)
                x = self.acti_gnn[i](x)
                if(self.dropout_rate > 0):
                    x = self.dropout_gnn[i](x)
        elif(self.wiring_mode=="m1"):
            for i in range(self.n_gnn_layers):
                x = self.GNN_layers[i](x, mask=x_mask)
                x = self.batch_norm[i](x)
                x = self.acti_gnn[i](x)
                if(self.dropout_rate > 0):
                    x = self.dropout_gnn[i](x)
        elif(self.wiring_mode=="m3"):
            for i in range(self.n_gnn_layers):
                if(i%3==0 and i > 0):
                    x = tf.concat([x0, x], axis=2)
                x = self.GNN_layers[i](x, mask=x_mask)
                if(i%3==0):
                    x0 = x
                x = self.batch_norm[i](x)
                x = self.acti_gnn[i](x)
                if(self.dropout_rate > 0):
                    x = self.dropout_gnn[i](x)


        if("p4" in self.mode):
            xx_p4 = xx[:,:,self.map_features['pfCand_px']:self.map_features['pfCand_E']+1]
            xx_p4_shape = tf.shape(xx_p4)
            xx_p4_other = xx[:,:,self.map_features['pfCand_pt']:self.map_features['pfCand_mass']+1]

            x_coor = x[:,:, -self.n_dim_gnn:]
            x_coor = tf.math.square(x_coor)
            d = tf.square(tf.math.reduce_sum(x_coor, axis = -1))
            w = tf.reshape(tf.math.exp(-10*d), (xx_p4_shape[0], xx_p4_shape[1], 1))

            x_mask_shape = tf.shape(x_mask)
            x_mask = tf.reshape(x_mask, (x_mask_shape[0], x_mask_shape[1], 1))
            sum_p4 = tf.reduce_sum(xx_p4 * w * x_mask, axis=1)
            # print('sum_p4.shape: ', sum_p4.shape) #(100,4)
            sum_p4_other = self.ToPtM2(sum_p4)

            x = tf.concat([x, xx_p4, xx_p4_other], axis = 2)

            #xx_p4 = tf.reshape(xx_p4, (xx_p4_shape[0], xx_p4_shape[1] * xx_p4_shape[2]))
            x_shape = tf.shape(x)
            x = tf.reshape(x, (x_shape[0], x_shape[1] * x_shape[2]))
            x = tf.concat([x, sum_p4, sum_p4_other], axis = 1)
            
        elif("dm"==self.mode):
            x_shape = tf.shape(x)
            x = tf.reshape(x, (x_shape[0], x_shape[1] * x_shape[2]))


        for i in range(self.n_dense_layers-1):
            x = self.dense[i](x)
            x = self.dense_batch_norm[i](x)
            x = self.dense_acti[i](x)
            if(self.dropout_rate > 0):
                x = self.dropout_dense[i](x)
        x = self.dense2(x)

        x_zeros = tf.zeros((x_shape[0], 2))
        if(self.mode == "dm"):
            xout = tf.concat([x, x_zeros], axis=1)
        elif self.mode == "p4":
            xout = tf.concat([x_zeros, x], axis=1)
        else:
            xout = x

        # print('xout shape: ',xout)
        return xout

    def ToPtM2(self, x):
        mypx  = x[:,0]
        mypy  = x[:,1]
        mypz  = x[:,2]
        myE   = x[:,3]

        mypx2  = tf.square(mypx)
        mypy2  = tf.square(mypy)
        mypz2  = tf.square(mypz)
        myE2   = tf.square(myE)

        mypt   = tf.sqrt(mypx2 + mypy2)
        mymass = myE2 - mypx2 - mypy2 - mypz2
        absp   = tf.sqrt(mypx2 + mypy2 + mypz2)

        return tf.stack([mypt,mymass], axis=1)


### Function that creates generators:
def make_generator(entry_begin, entry_end, z = False):
    _data_loader = R.DataLoader(n_tau, entry_begin, entry_end)
    _n_batches   = _data_loader.NumberOfBatches()

    def _generator():
        cnt = 0
        while True:
            _data_loader.Reset()
            while _data_loader.HasNext():
                data = _data_loader.LoadNext()
                x_np = np.asarray(data.x)
                x_3d = x_np.reshape((n_tau, n_pf, n_fe))
                y_np = np.asarray(data.y)
                y_2d = y_np.reshape((n_tau, n_labels))
                if z == True:
                    z_np = np.asarray(data.z)
                    z_2d = z_np.reshape((n_tau, n_labels-1))
                    yield x_3d, y_2d, z_2d
                else:
                    yield x_3d, y_2d
            ++cnt
            if cnt == 100:
                gc.collect() # garbage collection to improve preformance
                cnt = 0

    return _generator, _n_batches

#############################################################################################
##### Custom metrics:
### Accuracy calculation for number of charged/neutral hadrons:
@tf.function
def my_acc(y_true, y_pred):
    # print('\naccuracy calcualtion:')
    y_true = tf.math.round(y_true)
    y_true_int = tf.cast(y_true, tf.int32)
    y_pred = tf.math.round(y_pred)
    y_pred_int = tf.cast(y_pred, tf.int32)
    result = tf.math.logical_and(y_true_int[:, 0] == y_pred_int[:, 0], y_true_int[:, 1] == y_pred_int[:, 1])
    return tf.cast(result, tf.float32)

@tf.function
def my_mse_ch(y_true, y_pred):
    def_mse = 0.19802300936720527
    w = 1/def_mse
    return w*tf.square(y_true[:,0] - y_pred[:,0])

@tf.function
def my_mse_neu(y_true, y_pred):
    def_mse = 0.4980008353282306
    w = 1/def_mse
    return w*tf.square(y_true[:,1] - y_pred[:,1])

@tf.function
def my_mse_pt(y_true, y_pred):
    def_mse = 0.022759110487849007 # relative
    w = 1/def_mse
    return w*tf.square((y_true[:,2] - y_pred[:,2]) / y_true[:,2])

@tf.function
def my_mse_mass(y_true, y_pred):
    def_mse = 0.5968616152311431
    w = 1/def_mse
    return w*tf.square(y_true[:,3] - y_pred[:,3])


### Resolution of 4-momentum:
class MyResolution(tf.keras.metrics.Metric):
    def __init__(self, _name, var_pos, is_relative = False,**kwargs):
        super(MyResolution, self).__init__(name=_name,**kwargs)
        self.is_relative = is_relative
        self.var_pos     = var_pos
        self.sum_x       = self.add_weight(name="sum_x", initializer="zeros")
        self.sum_x2      = self.add_weight(name="sum_x2", initializer="zeros")
        self.N           = self.add_weight(name="N", initializer="zeros")

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.N.assign_add(tf.cast(tf.shape(y_true)[0], dtype=tf.float32))
        if(self.is_relative):
            self.sum_x.assign_add(tf.math.reduce_sum((y_pred[:,self.var_pos]   - y_true[:,self.var_pos])/y_true[:,self.var_pos]))
            self.sum_x2.assign_add(tf.math.reduce_sum(((y_pred[:,self.var_pos] - y_true[:,self.var_pos])/y_true[:,self.var_pos])**2))
        else:
            self.sum_x.assign_add(tf.math.reduce_sum(y_pred[:,self.var_pos]   - y_true[:,self.var_pos]))
            self.sum_x2.assign_add(tf.math.reduce_sum((y_pred[:,self.var_pos] - y_true[:,self.var_pos])**2))

    @tf.function
    def result(self):
        mean_x  = self.sum_x/self.N
        mean_x2 = self.sum_x2/self.N
        return mean_x2 -  mean_x**2

    def reset(self):
        self.sum_x.assign(0.)
        self.sum_x2.assign(0.)
        self.N.assign(0.)

    def get_config(self):
        config = {
            "is_relative": self.is_relative,
            "var_pos": self.var_pos,
            "name": self.name,
        }
        base_config = super(MyResolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(config):
        raise RuntimeError("Im here")
        return MyResolution(config["name"], config["var_pos"], is_relative=config["is_relative"])

pt_res_obj_rel = MyResolution('pt_res_rel' , 2 ,True)
pt_res_obj     = MyResolution('pt_res' , 2 ,False)
m2_res_obj     = MyResolution('m^2_res', 3 ,False)

def pt_res(y_true, y_pred, sample_weight=None):
    global pt_res_obj
    pt_res_obj.update_state(y_true, y_pred)
    return pt_res_obj.result()

def pt_res_rel(y_true, y_pred, sample_weight=None):
    global pt_res_obj_rel
    pt_res_obj_rel.update_state(y_true, y_pred)
    return pt_res_obj_rel.result()

def m2_res(y_true, y_pred, sample_weight=None):
    global m2_res_obj
    m2_res_obj.update_state(y_true, y_pred)
    return m2_res_obj.result()

def decay_mode_histo(x1, x2, dm_bins):
    decay_mode = np.zeros((x1.shape[0],2))
    decay_mode[:,0] = x1
    decay_mode[:,1] = x2
    h_dm, _ = np.histogramdd(decay_mode, bins=[dm_bins,dm_bins])
    h_dm[:,-1] = h_dm[:,4]+h_dm[:,-1] # sum the last and 4. column into the last column
    h_dm = np.delete(h_dm,4,1)        # delete the 4. column
    h_dm[-1,:] = h_dm[4,:]+h_dm[-1,:] # sum the last and 4. column into the last column
    h_dm = np.delete(h_dm,4,0)        # delete the 4. column
    return h_dm

##### Custom loss function:
class CustomMSE(keras.losses.Loss):
    mode = None

    def __init__(self, name="custom_mse", **kwargs):
        super().__init__(name=name,**kwargs)

    @tf.function
    def call(self, y_true, y_pred):
        y_shape = tf.shape(y_true)
        mse = tf.zeros(y_shape[0])
        if "dm" in CustomMSE.mode:
            mse = my_mse_ch(y_true, y_pred) + my_mse_neu(y_true, y_pred)
        if "p4" in CustomMSE.mode:
            mse = mse + my_mse_pt(y_true, y_pred) + my_mse_mass(y_true, y_pred)
        return mse


class ValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.generator_val, self.n_batches_val = make_generator(entry_start_val, entry_stop_val)
        self.dataset = tf.data.Dataset.from_generator(self.generator_val,(tf.float32, tf.float32),\
                            (tf.TensorShape([None,n_pf,n_fe]), tf.TensorShape([None,4])))

    def on_epoch_begin(self, epoch, logs=None):
        ### Reset the variables of class MyResolution:
        global pt_res_obj
        global pt_res_obj_rel
        global m2_res_obj
        pt_res_obj.reset()
        pt_res_obj_rel.reset()
        m2_res_obj.reset()
        print('Resolution reset done\n')

    def on_epoch_end(self, epoch, logs=None):
        # keys = list(logs.keys())
        # print("Log keys: {}".format(keys))
        print("\nValidation:")

        myresults = self.model.evaluate(x = self.dataset, batch_size = self.n_batches_val, steps = n_steps_val, verbose=2)
        if len(self.model.metrics_names) == 1:
            i = self.model.metrics_names[0]
            logs["val_"+i] = myresults
        else:
            cnt = 0
            for i in self.model.metrics_names:
                logs["val_"+i] = myresults[cnt]
                cnt += 1

        ### Save the entire models:
        self.model.save(os.path.join(self.model.filename,"my_model_{}".format(epoch+1)),save_format='tf')
        print('Model is saved.')
