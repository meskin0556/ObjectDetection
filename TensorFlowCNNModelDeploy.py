import tensorflow as tf


with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('model/TensorFlowCNNDemo3.ckpt.meta')
    new_saver.restore(sess,'model/TensorFlowCNNDemo3.ckpt' )
    graph = tf.get_default_graph()
    x = graph.get_operation_by_name('x').outputs[0]
    y = tf.get_collection("pred_network")[0]
    print("109的预测值是:", sess.run(y, feed_dict={x: [[109]]}))