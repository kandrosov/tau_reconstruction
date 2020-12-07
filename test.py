import tensorflow as tf

ccss1 = tf.constant([True,True,False])
print('ccss1: ',ccss1)
test1 = tf.constant([True])
test2 = tf.math.reduce_all(ccss1)
print('test2: ', test2)
if(test2==test1): 
    print('hi true')
else:
    print('hi false')

test_count = tf.math.count_nonzero(ccss1)
print('count non zeros: ', test_count)

test = 0
if(test == 0):
    print('works!!!')
else:
    print('does not work !!!')
